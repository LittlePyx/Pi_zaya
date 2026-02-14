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
from .geometry_utils import _rect_area, _union_rect, _rect_intersection_area
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
    _page_has_references_heading,
    _page_looks_like_references_content,
    detect_header_tag,
)
from .tables import _extract_tables_by_layout, _is_markdown_table_sane, table_text_to_markdown
from .block_classifier import _looks_like_math_block, _looks_like_code_block
from .llm_worker import LLMWorker
from .post_processing import postprocess_markdown
from .md_analyzer import MarkdownAnalyzer


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
        speed_mode = getattr(self.cfg, 'speed_mode', 'balanced')
        speed_config = self._get_speed_mode_config(speed_mode, total_pages)
        self._active_speed_config = speed_config
        
        # Use LLM config from cfg (already set by test2.py)
        use_llm = False
        llm_config = self.cfg.llm
        
        # If LLM config is provided, use it
        if llm_config:
            use_llm = True
            print(f"Using LLM ({llm_config.model}) for processing", flush=True)
        else:
            print("LLM not configured, using fast mode", flush=True)
        
        if use_llm and llm_config:
            # LLM mode might run in parallel if configured
            md_pages = self._process_batch_llm(doc, pdf_path, assets_dir)
        else:
            # Fast mode
            md_pages = self._process_batch_fast(doc, pdf_path, assets_dir)
            
        final_md = "\n\n".join(filter(None, md_pages))
        final_md = postprocess_markdown(final_md)
        
        # Post-process: merge split formulas and fix structure
        final_md = self._merge_split_formulas(final_md)
        final_md = self._fix_heading_structure(final_md)
        
        # Light LLM cleanup - only for remaining issues
        # Most fixes should be done during page processing
        
        if speed_config.get('enable_final_llm_cleanup', True) and (
            (use_llm and llm_config) or (self.cfg.llm and hasattr(self.llm_worker, '_client') and self.llm_worker._client)
        ):
            print(f"Running final LLM cleanup pass (speed mode: {speed_mode})...")
            import time
            start_time = time.time()
            
            # Quick checks to skip unnecessary processing
            has_inline_formulas = '$' in final_md and not final_md.count('$$') == final_md.count('$') // 2
            has_display_math = '$$' in final_md
            
            # Check references more carefully - need to check if first line after References starts with [number]
            has_references = False
            if "# References" in final_md or "## References" in final_md:
                lines_check = final_md.splitlines()
                for i, line in enumerate(lines_check):
                    if re.match(r'^#+\s+References', line, re.IGNORECASE):
                        # Check first non-empty line after References
                        for j in range(i+1, min(i+10, len(lines_check))):
                            next_line = lines_check[j].strip()
                            if next_line:
                                if re.match(r'^\[\d+\]', next_line):
                                    # First line is formatted, check a few more to be sure
                                    formatted_count = 0
                                    total_count = 0
                                    for k in range(j, min(j+20, len(lines_check))):
                                        check_line = lines_check[k].strip()
                                        if check_line:
                                            total_count += 1
                                            if re.match(r'^\[\d+\]', check_line):
                                                formatted_count += 1
                                    # If 80%+ are formatted, skip
                                    if total_count > 0 and formatted_count / total_count >= 0.8:
                                        has_references = False  # Already formatted
                                    else:
                                        has_references = True  # Needs formatting
                                else:
                                    has_references = True  # Needs formatting
                                break
                        break
            
            # Check tables - need proper format with --- separator
            has_tables = False
            if '|' in final_md:
                # Check if tables have proper separator
                lines_check = final_md.splitlines()
                for i, line in enumerate(lines_check):
                    stripped = line.strip()
                    if stripped.startswith('|') and '|' in stripped[1:] and len(stripped) > 10:
                        # Check if next few lines have --- separator
                        has_separator = False
                        for j in range(i+1, min(i+5, len(lines_check))):
                            if '---' in lines_check[j] or '|' in lines_check[j]:
                                if '---' in lines_check[j]:
                                    has_separator = True
                                break
                        if not has_separator:
                            has_tables = True  # Needs formatting
                            break
            mojibake_patterns = ['ďŹ', 'Ď', 'Î´', 'Îą', 'â']
            has_mojibake = any(p in final_md for p in mojibake_patterns)
            
            # Parallelize independent LLM operations for speed
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import threading
            
            # Create a lock for thread-safe final_md updates
            md_lock = threading.Lock()
            
            def safe_update_md(update_func, *args):
                """Thread-safe markdown update"""
                with md_lock:
                    nonlocal final_md
                    final_md = update_func(final_md, *args)
            
            # Prepare parallel tasks
            tasks = []
            
            # Task 1: Fix misclassified headings (must be first, others may depend on it)
            tasks.append(('headings', lambda: self._llm_fix_misclassified_headings(final_md)))
            
            # Task 2-4: Can run in parallel after headings are fixed (based on speed mode)
            if speed_config['fix_inline_formulas'] and has_inline_formulas:
                tasks.append(('inline_formulas', lambda: self._llm_fix_inline_formulas(final_md)))
            if speed_config['fix_display_math'] and has_display_math:
                tasks.append(('display_math', lambda: self._llm_fix_display_math(final_md)))
            if speed_config['fix_mojibake'] and has_mojibake:
                tasks.append(('mojibake', lambda: self._llm_light_cleanup(final_md)))
            
            # Process headings first (sequential) if enabled
            if speed_config['fix_headings'] and tasks:
                task_name, task_func = tasks[0]
                final_md = task_func()
                tasks = tasks[1:]
            elif not speed_config['fix_headings']:
                tasks = []  # Skip headings if disabled
            
            # Process remaining tasks in parallel (based on speed mode)
            max_workers = speed_config['max_parallel_llm_tasks']
            if tasks:
                print(f"Processing {len(tasks)} tasks in parallel (max {max_workers} workers)...")
                with ThreadPoolExecutor(max_workers=min(max_workers, len(tasks))) as executor:
                    futures = {executor.submit(task_func): task_name for task_name, task_func in tasks}
                    task_results = {}
                    for future in as_completed(futures):
                        task_name = futures[future]
                        try:
                            result = future.result(timeout=speed_config['timeout_per_task'])
                            task_results[task_name] = result
                        except Exception as e:
                            print(f"Task {task_name} failed: {e}")
                            task_results[task_name] = None
                    
                    # Apply results: tasks run in parallel on same input, apply in priority order
                    # Priority: mojibake (affects all text) -> inline_formulas -> display_math
                    with md_lock:
                        # Apply fixes in order of dependency
                        # Start with the base (headings already fixed)
                        current_md = final_md
                        
                        # Apply mojibake fix first (affects all text including formulas)
                        if 'mojibake' in task_results and task_results['mojibake']:
                            current_md = task_results['mojibake']
                        
                        # Apply inline formulas fix (works on $...$ blocks, may include mojibake fixes)
                        if 'inline_formulas' in task_results and task_results['inline_formulas']:
                            # Inline formulas fix is based on original, but we want to apply it to mojibake-fixed version
                            # Since the fix is specific to $...$ blocks, we can merge by replacing inline formulas
                            # For simplicity, if both exist, prefer inline_formulas (it may have better formula fixes)
                            if task_results['inline_formulas']:
                                # Check if mojibake was also applied - if so, we need to merge
                                if 'mojibake' in task_results and task_results['mojibake']:
                                    # Merge: use inline_formulas for $...$ blocks, mojibake for rest
                                    # Extract inline formulas from inline_formulas result
                                    inline_formulas_fixed = task_results['inline_formulas']
                                    inline_pattern = r'\$[^$]+\$'
                                    inline_matches = list(re.finditer(inline_pattern, inline_formulas_fixed))
                                    if inline_matches:
                                        # Replace inline formulas in current_md
                                        current_md_parts = []
                                        last_end = 0
                                        for match in inline_matches:
                                            current_md_parts.append(current_md[last_end:match.start()])
                                            current_md_parts.append(match.group())
                                            last_end = match.end()
                                        current_md_parts.append(current_md[last_end:])
                                        current_md = ''.join(current_md_parts)
                                    else:
                                        current_md = inline_formulas_fixed
                                else:
                                    current_md = task_results['inline_formulas']
                        
                        # Apply display math fix (works on $$...$$ blocks)
                        if 'display_math' in task_results and task_results['display_math']:
                            # Display math fix is specific to $$...$$ blocks
                            display_math_fixed = task_results['display_math']
                            # Extract display math blocks from fixed version
                            display_pattern = r'\$\$.*?\$\$'
                            display_blocks_fixed = re.findall(display_pattern, display_math_fixed, re.DOTALL)
                            if display_blocks_fixed:
                                # Replace display math blocks in current_md
                                current_md_parts = re.split(display_pattern, current_md, flags=re.DOTALL)
                                display_blocks_original = re.findall(display_pattern, current_md, re.DOTALL)
                                if len(display_blocks_fixed) == len(display_blocks_original):
                                    # Replace each block
                                    result_parts = [current_md_parts[0]]
                                    for i, fixed_block in enumerate(display_blocks_fixed):
                                        result_parts.append(fixed_block)
                                        if i + 1 < len(current_md_parts):
                                            result_parts.append(current_md_parts[i + 1])
                                    current_md = ''.join(result_parts)
                                else:
                                    # Count mismatch, use fixed version for display math sections
                                    current_md = display_math_fixed
                            else:
                                # No display math blocks, keep current
                                pass
                        
                        final_md = current_md
            
            # Sequential tasks (depend on previous results) - only if enabled in speed mode
            # Fourth, fix references formatting (only if enabled and needed)
            if speed_config.get('fix_references', False) and has_references:
                if speed_config.get('use_crossref', False):
                    final_md = self._llm_fix_references_with_crossref(final_md, save_dir)
                else:
                    final_md = self._llm_fix_references(final_md)
            elif has_references:
                if not speed_config.get('fix_references', True):
                    print("References formatting skipped (speed mode)")
                else:
                    print("References already formatted, skipping...")
            
            # Fifth, fix tables formatting (only if enabled and needed)
            if speed_config.get('fix_tables', False) and has_tables:
                if speed_config.get('use_table_screenshot', False):
                    final_md = self._llm_fix_tables_with_screenshot(final_md, pdf_path, save_dir)
                else:
                    final_md = self._llm_fix_tables(final_md)
            elif has_tables:
                if not speed_config.get('fix_tables', True):
                    print("Tables formatting skipped (speed mode)")
                else:
                    print("Tables already formatted, skipping...")
            
            # For full_llm mode, add a final comprehensive quality check pass
            if speed_mode == 'full_llm':
                print("Running final comprehensive quality check (full_llm mode)...")
                final_md = self._llm_final_quality_check(final_md)
            
            elapsed = time.time() - start_time
            print(f"LLM cleanup completed in {elapsed:.1f}s")
        
        out_file = save_dir / "output.md"
        out_file.write_text(final_md, encoding="utf-8")
        print(f"Saved to {out_file}", flush=True)
        print(f"Conversion completed successfully!", flush=True)
        
        # Analyze quality and generate report
        if self.analyze_quality:
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

    def _process_batch_fast(self, doc, pdf_path: Path, assets_dir: Path) -> List[Optional[str]]:
        """Process pages sequentially."""
        results = [None] * len(doc)
        total_pages = len(doc)
        for i, page in enumerate(doc):
            print(f"Processing page {i+1}/{total_pages} ...", flush=True)
            results[i] = self._process_page(
                page, 
                page_index=i, 
                pdf_path=pdf_path, 
                assets_dir=assets_dir
            )
            print(f"Finished page {i+1}/{total_pages}", flush=True)
        return results

    def _process_batch_llm(self, doc, pdf_path: Path, assets_dir: Path) -> List[Optional[str]]:
        """Process pages in parallel with LLM support."""
        total_pages = len(doc)
        results = [None] * total_pages
        
        # Determine number of workers - optimize based on speed mode
        import os
        import multiprocessing
        speed_mode = getattr(self.cfg, 'speed_mode', 'balanced')
        speed_config = self._get_speed_mode_config(speed_mode, total_pages)
        
        num_workers = int(os.environ.get("KB_PDF_WORKERS", "0"))
        if num_workers <= 0:
            # Use speed mode configuration
            max_parallel = speed_config['max_parallel_pages']
            cpu_count = multiprocessing.cpu_count()
            if total_pages <= 2:
                num_workers = 1
            else:
                num_workers = min(max_parallel, cpu_count, total_pages)
        
        if num_workers <= 1 or total_pages <= 1:
            # Sequential processing
            return self._process_batch_fast(doc, pdf_path, assets_dir)
        
        # Parallel processing with timeout protection
        print(f"Processing {total_pages} pages in parallel with {num_workers} workers...", flush=True)
        from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
        
        def process_single_page(args):
            i, page = args
            try:
                # Output processing message before starting
                print(f"Processing page {i+1}/{total_pages} ...", flush=True)
                # Add timeout protection (5 minutes per page)
                result = self._process_page(
                    page,
                    page_index=i,
                    pdf_path=pdf_path,
                    assets_dir=assets_dir
                )
                return i, result
            except Exception as e:
                print(f"Error processing page {i+1}: {e}", flush=True)
                import traceback
                traceback.print_exc()
                return i, None
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(process_single_page, (i, doc[i])): i
                for i in range(total_pages)
            }
            
            completed = 0
            done_pages = set()
            for future in as_completed(futures):
                try:
                    i, result = future.result(timeout=300)  # 5 minute timeout per page
                    results[i] = result
                    done_pages.add(i + 1)  # Page numbers are 1-indexed
                    completed = len(done_pages)
                    # Output progress in expected format
                    print(f"Finished page {i+1}/{total_pages}", flush=True)
                except Exception as e:
                    i = futures[future]
                    error_msg = str(e)
                    if "timeout" in error_msg.lower() or "Timeout" in error_msg:
                        print(f"\nTimeout processing page {i+1} (exceeded 5 minutes)", flush=True)
                    else:
                        print(f"\nError processing page {i+1}: {e}", flush=True)
                    results[i] = None
        return results

    def _process_page(self, page, page_index: int, pdf_path: Path, assets_dir: Path) -> str:
        # 1. Analyze Layout
        body_size = detect_body_font_size([page]) # Heuristic was on full doc, but per-page is okay fallback
        
        # 2. Check if this is a references page (for special handling)
        is_references_page = _page_has_references_heading(page) or _page_looks_like_references_content(page)
        
        # 3. Extract specific rects (excluding header/footer regions)
        W = float(page.rect.width)
        H = float(page.rect.height)
        header_threshold = H * 0.12
        footer_threshold = H * 0.88
        visual_rects = _collect_visual_rects(page)
        # Filter visual rects in header/footer (unless they're large figures)
        visual_rects = [
            r for r in visual_rects 
            if not (r.y1 < header_threshold or r.y0 > footer_threshold) or _rect_area(r) > (W * H * 0.15)
        ]
        
        # 4. Extract Tables
        # We need to map table rects to avoid processing them as text
        tables_found = _extract_tables_by_layout(
            page, 
            pdf_path=pdf_path, 
            page_index=page_index,
            visual_rects=visual_rects
        )
        
        # 5. Extract Text Blocks
        blocks = self._extract_text_blocks(
            page, 
            page_index=page_index, 
            body_size=body_size,
            tables=tables_found,
            visual_rects=visual_rects,
            assets_dir=assets_dir,
            is_references_page=is_references_page
        )
        
        # 6. LLM Classification / Repair
        speed_cfg = getattr(self, "_active_speed_config", None) or {}
        if self.cfg.llm and speed_cfg.get("use_llm_for_all", True):
            # Enhance blocks with LLM
            blocks = self._enhance_blocks_with_llm(blocks, page_index, page)
            
        # 7. Render
        return self._render_blocks_to_markdown(blocks, page_index)

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
            
            # Join lines more carefully - preserve line breaks for formulas
            # For formulas, we want to keep line structure
            full_text = " ".join(lines_text).strip()
            if not full_text:
                continue
            
            # Check if this looks like author information (should not be heading)
            # Pattern: "Name 1 , Name 2 , Name 3 , and Name 4 ,"
            import re
            if re.search(r'^\s*[A-Z][a-z]+\s+\d+\s*,.*\d+\s*,.*and.*\d+\s*,?\s*$', full_text):
                # This is likely author information, not a heading
                is_caption = False  # Ensure it's not classified as caption
                is_math = False
                heading_level = None
                # Create as regular text block
                tb = TextBlock(
                    bbox=tuple(bbox),
                    text=full_text,
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
            
            if len(lines_text) >= 1:
                # Check if this is a caption (Figure/Table caption)
                from .heuristics import _is_caption_like_text
                is_caption = _is_caption_like_text(full_text)
                
                # Check if this block looks like a math formula (but not if it's a caption)
                if not is_caption:
                    is_math = _looks_like_math_block(lines_text)
                    # Check if this block looks like code (only if not math)
                    if not is_math:
                        is_code = _looks_like_code_block(lines_text)
            
            # Filter noise - check if in header/footer region
            is_header_footer = False
            header_threshold = H * 0.12  # Top 12% of page
            footer_threshold = H * 0.88  # Bottom 12% of page
            if bbox.y1 < header_threshold or bbox.y0 > footer_threshold:
                is_header_footer = True
                # Still allow if it's a major heading (e.g., section title at top of page)
                if not (max_size > body_size + 1.5 and is_bold and len(full_text) < 100):
                    if full_text in self.noise_texts or _is_noise_line(full_text):
                        continue
            
            if full_text in self.noise_texts:
                continue
            if _is_frontmatter_noise_line(full_text):
                continue
            if _is_noise_line(full_text) and not is_header_footer:
                continue
                
            # Detect heading (heuristic-based, LLM will refine later if available)
            # Skip heading detection if this is already classified as math, code, or caption
            heading_level = None
            if not is_math and not is_code and not is_caption:
                # Additional check: very short text with numbers is likely formula, not heading
                import re
                text_stripped = full_text.strip()
                if len(text_stripped) <= 25:
                    # Check for formula-like patterns more strictly
                    if re.search(r'^\s*[A-Z]?\s*\d+\s*[a-z]', text_stripped) or \
                       re.search(r'^\s*\d+\s*[a-z]+\s*[+\-]', text_stripped) or \
                       re.search(r'^\s*[a-z]\s*\d+', text_stripped) or \
                       (re.search(r'\d+.*[a-z]|[a-z].*\d+', text_stripped) and not re.search(r'[A-Z]{2,}', text_stripped) and '=' not in text_stripped):
                        is_math = True  # Reclassify as math
                        is_caption = False
                
                if not is_math:
                    heading_tag = detect_header_tag(
                        page_index=page_index,
                        text=full_text,
                        max_size=max_size,
                        is_bold=is_bold,
                        body_size=body_size,
                        page_width=W,
                        bbox=tuple(bbox),
                    )
                    if heading_tag:
                        heading_level = heading_tag
                
            # Create Block with detected types
            tb = TextBlock(
                bbox=tuple(bbox),
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
        # Filter out header/footer regions and crop properly
        header_threshold = H * 0.12
        footer_threshold = H * 0.88
        side_margin = W * 0.05  # 5% margin on sides
        
        # Detect column layout for proper image handling
        col_split = _detect_column_split_x(text_blocks, page_width=W) if text_blocks else None
        spanning_threshold = W * 0.55  # Full-width images span both columns
        
        for rect in visual_rects:
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
            
            # Crop rect slightly to avoid edge artifacts (but preserve full-width images)
            crop_margin = 2.0 if not is_full_width else 1.0  # Less cropping for full-width
            cropped_rect = fitz.Rect(
                max(0, rect.x0 + crop_margin),
                max(0, rect.y0 + crop_margin),
                min(W, rect.x1 - crop_margin),
                min(H, rect.y1 - crop_margin)
            )
            
            if cropped_rect.width <= 0 or cropped_rect.height <= 0:
                continue
                
            # Save image with proper DPI from config
            img_name = f"page_{page_index+1}_fig_{int(rect.x0)}_{int(rect.y0)}.png"
            img_path = assets_dir / img_name
            
            # Use configured DPI
            dpi = self.dpi
            try:
                pix = page.get_pixmap(clip=cropped_rect, dpi=dpi)
                pix.save(img_path)
            except Exception as e:
                # Fallback to original rect if crop fails
                try:
                    pix = page.get_pixmap(clip=rect, dpi=dpi)
                    pix.save(img_path)
                except Exception:
                    continue
            
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
            
        # Apply classifications - create new blocks since TextBlock is immutable
        enhanced_blocks = []
        for i, b in enumerate(blocks):
            # Find classification for this block
            item = None
            for cls_item in classified:
                if cls_item.get("i") == i:
                    item = cls_item
                    break
            
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
        
        # Normalize text but preserve line structure for multi-line formulas
        lines = text.splitlines()
        normalized_lines = []
        for line in lines:
            t = _normalize_text(line)
            if t:
                normalized_lines.append(t)
        
        if not normalized_lines:
            return ""
        
        t = " ".join(normalized_lines)
        
        # Remove equation numbers at the end like "(8)" or "(EQNO 8)"
        t = re.sub(r'\(\s*(?:EQNO\s+)?\d+\s*\)\s*$', '', t).strip()
        
        # Replace Greek letters first (before other processing)
        for greek, latex in GREEK_TO_LATEX.items():
            t = t.replace(greek, latex)
        
        # Replace math symbols
        for symbol, latex in MATH_SYMBOL_TO_LATEX.items():
            t = t.replace(symbol, latex)
        
        # Normalize whitespace but be careful with subscripts
        # First, protect potential subscripts/superscripts
        # Pattern: letter followed by space and then digit or lowercase letter
        # But only if not already in LaTeX format
        
        # Fix common math functions
        t = re.sub(r'\blog\b', r'\\log', t)
        t = re.sub(r'\bexp\b', r'\\exp', t)
        t = re.sub(r'\bsin\b', r'\\sin', t)
        t = re.sub(r'\bcos\b', r'\\cos', t)
        t = re.sub(r'\btan\b', r'\\tan', t)
        t = re.sub(r'\bmax\b', r'\\max', t)
        t = re.sub(r'\bmin\b', r'\\min', t)
        t = re.sub(r'\bln\b', r'\\ln', t)
        t = re.sub(r'\bsqrt\b', r'\\sqrt', t)
        
        # Fix subscripts more carefully
        # Pattern: variable name (single letter or word) followed by space and digit/single letter
        # Only if it's not already in LaTeX format
        if '_{' not in t and '_' not in t:
            # Simple case: "x 1" -> "x_1", but be careful
            # Only do this for single letters followed by single digits/letters
            t = re.sub(r'\b([a-zA-Z])\s+(\d+)\b', r'\1_{\2}', t)
            # For single lowercase letters as subscripts: "x i" -> "x_i" (but not "x in")
            t = re.sub(r'\b([a-zA-Z])\s+([a-z])\b(?!\w)', r'\1_{\2}', t)
        
        # Normalize remaining whitespace
        t = re.sub(r'\s+', ' ', t).strip()
        
        # Fix common operators that might have been split
        t = re.sub(r'\s*=\s*', ' = ', t)
        t = re.sub(r'\s*\+\s*', ' + ', t)
        t = re.sub(r'\s*-\s*', ' - ', t)
        # Replace * with \cdot, but escape the backslash properly
        t = re.sub(r'\s*\*\s*', r' \\cdot ', t)
        t = re.sub(r'\s*/\s*', ' / ', t)
        
        # Clean up extra spaces around operators
        t = re.sub(r'\s+', ' ', t)
        
        return t

    def _render_blocks_to_markdown(self, blocks: List[TextBlock], page_index: int) -> str:
        out = []
        for b in blocks:
            # Check if this is an image block (images are stored as text blocks with markdown image syntax)
            if b.text and (b.text.startswith("![") or re.match(r'^!\[.*?\]\(.*?\)', b.text)):
                # This is an image block - output it directly
                out.append(b.text)
                out.append("")  # Add blank line after image
                continue
            
            if b.heading_level:
                # [H1] -> #
                lvl = int(b.heading_level.replace("[H", "").replace("]", ""))
                heading_text = b.text.strip()
                
                # Use LLM to confirm heading and determine level
                llm_result = None
                if self.cfg.llm and hasattr(self.llm_worker, '_client') and self.llm_worker._client:
                    try:
                        llm_result = self.llm_worker.call_llm_confirm_and_level_heading(
                            heading_text,
                            page_number=page_index,
                            suggested_level=lvl
                        )
                    except Exception:
                        pass
                
                if llm_result and llm_result.get('is_heading'):
                    # LLM confirmed it's a heading
                    heading_text = llm_result.get('text', heading_text)
                    llm_level = llm_result.get('level')
                    
                    # Use LLM-determined level with proper mapping (user's requirement):
                    # "1. Introduction" -> Level 1 -> Markdown # (H1)
                    # "3.1. Background" -> Level 2 -> Markdown ## (H2)
                    # "3.1.1. Details" -> Level 3 -> Markdown ### (H3)
                    if llm_level:
                        lvl = llm_level
                    
                    # Normalize heading text for duplicate detection
                    normalized_heading = re.sub(r'^[IVX]+\.\s*', '', heading_text, flags=re.IGNORECASE).strip()
                    normalized_heading = re.sub(r'^[A-Z]\.\s*', '', normalized_heading).strip()
                    normalized_heading = re.sub(r'^\d+\.\s*', '', normalized_heading).strip()
                    
                    # Check for duplicates
                    if normalized_heading not in self.seen_headings:
                        # Update heading stack
                        while self.heading_stack and self.heading_stack[-1][0] >= lvl:
                            self.heading_stack.pop()
                        
                        self.heading_stack.append((lvl, heading_text))
                        self.seen_headings.add(normalized_heading)
                        out.append("#" * lvl + " " + heading_text)
                    else:
                        # Duplicate heading, convert to text
                        out.append(heading_text)
                else:
                    # LLM said it's not a heading, or LLM unavailable - use strict heuristic checks
                    # Filter out obvious non-headings first
                    if '@' in heading_text or re.search(r'\b(?:university|dept|department|institute|email|zhejiang|westlake)\b', heading_text, re.IGNORECASE):
                        # Author/affiliation line - not a heading
                        out.append(heading_text)
                        continue
                    
                    if re.search(r'[↑↓]', heading_text) or re.search(r'\b(?:PSNR|SSIM|LPIPS)\b', heading_text, re.IGNORECASE):
                        # Table header - not a heading
                        out.append(heading_text)
                        continue
                    
                    # Quick heuristic: very short or math-like -> not heading
                    if len(heading_text) <= 5 or not re.search(r'[A-Za-z]{3,}', heading_text):
                        out.append(heading_text)
                        continue
                    
                    # Check for math symbols
                    math_symbol_count = len(re.findall(r'[+\-*/=^_{}\[\]()]', heading_text))
                    if math_symbol_count > 2:
                        out.append(heading_text)
                        continue
                    
                    # Normalize for duplicate check
                    normalized_heading = re.sub(r'^[IVX]+\.\s*', '', heading_text, flags=re.IGNORECASE).strip()
                    normalized_heading = re.sub(r'^[A-Z]\.\s*', '', normalized_heading).strip()
                    normalized_heading = re.sub(r'^\d+\.\s*', '', normalized_heading).strip()
                    
                    if normalized_heading in self.seen_headings:
                        out.append(heading_text)
                        continue
                    
                    # Use heuristic level determination based on numbering pattern (user's requirement)
                    numbered_match = re.match(r'^(\d+(?:\.\d+)*)\.?\s+', heading_text)
                    if numbered_match:
                        num_parts = numbered_match.group(1).split('.')
                        if len(num_parts) == 1:
                            lvl = 1  # "1. Introduction" -> # (H1)
                        elif len(num_parts) == 2:
                            lvl = 2  # "3.1. Background" -> ## (H2)
                        else:
                            lvl = 3  # "3.1.1. Details" -> ### (H3)
                    else:
                        # Letter headings: "A. Appendix" -> ##
                        letter_match = re.match(r'^[A-Z]\.\s+', heading_text)
                        if letter_match:
                            lvl = 2
                        else:
                            # Keep original level from detection, but cap at 3
                            lvl = min(3, lvl)
                    
                    # Update heading stack
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
                # CRITICAL FIX: Use LLM to check if this is actually a heading that was misclassified as math
                text_stripped = b.text.strip()
                
                # Use LLM to confirm if it's a heading (user wants perfect heading identification)
                llm_result = None
                if self.cfg.llm and hasattr(self.llm_worker, '_client') and self.llm_worker._client:
                    try:
                        llm_result = self.llm_worker.call_llm_confirm_and_level_heading(
                            text_stripped,
                            page_number=page_index
                        )
                    except Exception:
                        pass
                
                if llm_result and llm_result.get('is_heading'):
                    # LLM confirmed it's a heading - treat as heading with LLM-determined level
                    heading_text = llm_result.get('text', text_stripped)
                    llm_level = llm_result.get('level', 2)
                    
                    # Normalize for duplicate check
                    normalized_heading = re.sub(r'^[IVX]+\.\s*', '', heading_text, flags=re.IGNORECASE).strip()
                    normalized_heading = re.sub(r'^[A-Z]\.\s*', '', normalized_heading).strip()
                    normalized_heading = re.sub(r'^\d+\.\s*', '', normalized_heading).strip()
                    
                    if normalized_heading not in self.seen_headings:
                        # Update heading stack
                        while self.heading_stack and self.heading_stack[-1][0] >= llm_level:
                            self.heading_stack.pop()
                        
                        self.heading_stack.append((llm_level, heading_text))
                        self.seen_headings.add(normalized_heading)
                        out.append("#" * llm_level + " " + heading_text)
                        continue
                    else:
                        # Duplicate, convert to text
                        out.append(heading_text)
                        continue
                
                # LLM said it's not a heading, or LLM unavailable - proceed with math rendering
                # Quick heuristic: very short or no letters -> definitely math
                if len(text_stripped) <= 5 or not re.search(r'[A-Za-z]', text_stripped):
                    # Too short or no letters - definitely math
                    pass
                
                # It's actually math, proceed with math rendering
                # ALWAYS try LLM repair first for better quality (user said speed is OK)
                latex_text = None
                if self.cfg.llm and self.llm_worker._client:
                    try:
                        # First attempt: standard repair
                        repaired = self.llm_worker.call_llm_repair_math(
                            b.text,
                            page_number=page_index,
                            block_index=len(out)
                        )
                        if repaired:
                            latex_text = repaired
                        else:
                            # Second attempt: more aggressive repair for inline math
                            # Check if it looks like inline math (short, no line breaks)
                            if len(b.text.strip()) <= 50 and "\n" not in b.text:
                                # Try with a more specific prompt for inline math
                                prompt = f"""Convert this garbled inline math expression to proper LaTeX.
The expression is: {b.text}

Requirements:
- Use proper LaTeX syntax (e.g., \\hat{{C}} not ˆ C, C(r)^2 not C ( r ) 2)
- Remove extra spaces
- Fix subscripts and superscripts properly
- Return ONLY the LaTeX code without $ delimiters

LaTeX:"""
                                try:
                                    resp = self.llm_worker._llm_create(
                                        messages=[
                                            {"role": "system", "content": "You are a LaTeX math expert specializing in inline math expressions."},
                                            {"role": "user", "content": prompt}
                                        ],
                                        temperature=0.0,
                                        max_tokens=200,
                                    )
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
                
                # If LLM didn't help, use heuristic conversion
                if not latex_text:
                    latex_text = self._convert_formula_to_latex(b.text)
                
                if latex_text:
                    # Clean up latex_text: remove trailing commas, fix nested $, etc.
                    latex_text = latex_text.strip()
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
                    has_break = "\n" in b.text
                    has_equals = "=" in latex_text
                    has_complex = any(op in latex_text for op in ['\\sum', '\\int', '\\prod', '\\frac', '\\sqrt', '\\exp', '\\log'])
                    is_long = len(latex_text) > 60
                    
                    is_inline = not (has_break or has_equals or has_complex or is_long)
                    
                    if is_inline:
                        # Inline math: ALWAYS use LLM to polish for better quality (user requested)
                        if self.cfg.llm and self.llm_worker._client:
                            try:
                                # More aggressive LLM polish for inline math
                                polish_prompt = f"""Convert this inline math expression to proper LaTeX. The expression may contain garbled characters or incorrect formatting.

Input: {latex_text}

Requirements:
1. Fix garbled characters (e.g., "ˆ C" -> "\\hat{{C}}", "C ( r ) 2" -> "C(r)^2")
2. Use proper LaTeX syntax:
   - Subscripts: x_i not x i
   - Superscripts: x^2 not x 2
   - Functions: \\hat{{C}} not ˆ C
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
                    fallback_text = b.text.strip()
                    fallback_text = re.sub(r',\s*$', '', fallback_text)
                    fallback_text = fallback_text.replace('$', '')
                    out.append(f"$$\n{fallback_text}\n$$")
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
                    if any(pattern in text for pattern in ['ďŹ', 'Ď', 'Î´', 'Îą', 'â']):
                        try:
                            repaired = self.llm_worker.call_llm_repair_body_paragraph(
                                text,
                                page_number=page_index,
                                block_index=len(out)
                            )
                            if repaired:
                                text = repaired
                        except Exception:
                            pass
                out.append(text)
            out.append("")
        return "\n".join(out)

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
                        if len(next_stripped) < 50 and re.search(r'[+\-*/=<>≤≥∈∑∫αβγδεθλμπστφω]', next_stripped):
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
                   re.search(r'[αβγδεζηθικλμνξοπρστυφχψω]', text, re.IGNORECASE) or \
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
                    # But include if it has garbled chars like ˆ
                    has_garbled = any(c in formula_text for c in ['ˆ', ' ']) and '\\' not in formula_text
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
1. Fix garbled Unicode: "ˆ" -> "\\hat", "C ( r )" -> "C(r)", "C ( r ) 2" -> "C(r)^2"
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
3. Fix any garbled text and mojibake (e.g., "Miloš Hašan" not garbled)
4. Ensure proper spacing and punctuation
5. Keep all citation numbers exactly as they appear
6. Each reference on a separate line
7. Remove any duplicate references
8. Fix special characters properly (e.g., "®" not garbled)

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
        eq_number = 1  # Equation number counter
        
        print("Fixing display math blocks...")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == "$$":
                if in_display_math:
                    # End of display math block
                    math_text = "\n".join(math_lines)
                    # Skip if this looks like a table (has | or many spaces/numbers)
                    if '|' in math_text or (len(math_text) > 100 and re.search(r'\d+\.\d+', math_text) and re.search(r'PSNR|SSIM', math_text, re.IGNORECASE)):
                        # This is likely a table, not math - convert to table format
                        # Try to parse as table
                        table_data = math_text.split()
                        if len(table_data) > 10:
                            # Convert to markdown table
                            try:
                                # Use LLM to convert to proper table
                                table_prompt = f"""Convert this text into a proper Markdown table.

Input text:
{math_text}

Requirements:
1. Identify column headers (e.g., "CR = 16", "PSNR ↑", "SSIM ↑")
2. Create proper Markdown table with | separators
3. Each row should be on a separate line
4. Fix any garbled text

Return ONLY the Markdown table, no explanations:"""
                                
                                resp = self.llm_worker._llm_create(
                                    messages=[
                                        {"role": "system", "content": "You are an expert at converting text data into Markdown tables."},
                                        {"role": "user", "content": table_prompt}
                                    ],
                                    temperature=0.0,
                                    max_tokens=2000,
                                )
                                table_md = (resp.choices[0].message.content or "").strip()
                                if table_md.startswith("```"):
                                    table_md = re.sub(r'^```(?:\w+)?\n?', '', table_md)
                                    table_md = re.sub(r'\n?```$', '', table_md)
                                
                                if table_md and '|' in table_md:
                                    fixed_lines.append(table_md)
                                    fixed_count += 1
                                else:
                                    # Fallback: keep as text
                                    fixed_lines.append(math_text)
                            except Exception:
                                # Fallback: keep as text
                                fixed_lines.append(math_text)
                        else:
                            fixed_lines.append(math_text)
                        
                        in_display_math = False
                        math_lines = []
                        continue
                    
                    # Check if there are nested $ symbols or \[ \]
                    if '$' in math_text or '\\[' in math_text or '\\]' in math_text:
                        try:
                            prompt = f"""Fix this display math block. Remove any nested $ symbols and fix formatting.

Input:
{math_text}

Requirements:
1. Remove ALL $ symbols inside the math block (display math uses $$...$$, no $ inside)
2. Remove any \\[ or \\] symbols (use $$ only for display math)
3. Fix garbled characters (e.g., "ˆ" -> "\\hat", "C ( r )" -> "C(r)")
4. Use proper LaTeX syntax (e.g., "Z_{t} f" -> "\\int_{t_n}^{t_f}" where t_n and t_f are time bounds)
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
                            # Add equation number if not present
                            if not re.search(r'\((\d+)\)\s*$', cleaned_math.strip()):
                                cleaned_math = f"{cleaned_math}\t\t({eq_number})"
                                eq_number += 1
                            fixed_lines.append("$$")
                            fixed_lines.append(cleaned_math)
                            fixed_lines.append("$$")
                    else:
                        # Clean up: remove empty lines and fix formatting
                        cleaned_lines = [l for l in math_lines if l.strip()]
                        if cleaned_lines:
                            # Add equation number to the last line if not present
                            last_line = cleaned_lines[-1]
                            if not re.search(r'\((\d+)\)\s*$', last_line.strip()):
                                cleaned_lines[-1] = f"{last_line}\t\t({eq_number})"
                                eq_number += 1
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
        mojibake_patterns = ['ďŹ', 'Ď', 'Î´', 'Îą', 'âĽ', 'âĺ¤']
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
   - "ďŹ" -> "fi"
   - "Ď" -> "σ" or "τ"
   - "Î´" -> "δ"
   - "Îą" -> "α"
   - "â" -> correct symbol (—, ∈, ∥, ≤, ≥, etc.)
   - "ˆ" -> "\\hat" (in math context)

2. FIX INLINE FORMULAS (single $...$):
   - Fix garbled math: "ˆ C ( r ) - C ( r ) 2" -> "\\hat{{C}}(r) - C(r)^2"
   - Remove extra spaces: "C ( r )" -> "C(r)"
   - Fix subscripts/superscripts: "x 2" -> "x^2" or "x_2"
   - Use proper LaTeX: "ˆ" -> "\\hat", "α" -> "\\alpha"

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
   - "ďŹ" -> "fi" (e.g., "ďŹexible" -> "flexible", "ďŹrst" -> "first", "ďŹxed" -> "fixed")
   - "Ď" -> "σ" (sigma) when in math context, or "τ" (tau) when appropriate
   - "Î´" -> "δ" (delta)
   - "Îą" -> "α" (alpha)
   - "â" -> various symbols: "âĽ" -> "∥" (norm), "âĺ¤" -> "≤" (leq), "âĺ¥" -> "≥" (geq), "â" -> "—" (em dash), "â" -> "∈" (element of)
   - "Ă" -> "×" (times)
   - "Ě" -> "≠" (not equal) or other symbols
   - Fix ALL garbled characters systematically

2. FIX ALL FORMULAS (CRITICAL):
   - Convert ALL Unicode math symbols to LaTeX:
     * α, β, γ, δ, ε, θ, λ, μ, π, σ, τ, φ, ω -> \\alpha, \\beta, \\gamma, \\delta, \\epsilon, \\theta, \\lambda, \\mu, \\pi, \\sigma, \\tau, \\phi, \\omega
     * ≤ -> \\leq, ≥ -> \\geq, ≠ -> \\neq, ∈ -> \\in, ∉ -> \\notin
     * ∑ -> \\sum, ∏ -> \\prod, ∫ -> \\int, ∞ -> \\infty
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
   - "â" -> "—" (em dash) in text
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
   - Fix ALL Unicode math symbols to LaTeX (α -> \\alpha, ≤ -> \\leq, etc.)
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
   - Fix ALL mojibake/garbled text (ďŹ -> fi, Ď -> σ, Î´ -> δ, etc.)
   - Fix all ligatures and special characters
   - Ensure proper paragraph structure
   - Preserve mathematical notation in text
   - Fix em dashes: "â" -> "—"

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
        """Get configuration for speed mode."""
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        
        configs = {
            'full_llm': {
                # 全LLM模式：标题结构、行间行内公式、图片、表格、reference、正文都要非常完美
                'use_llm_for_all': True,
                'max_parallel_llm_tasks': 8,  # Maximum parallelization
                'max_parallel_pages': min(16, cpu_count * 2),
                'fix_inline_formulas': True,  # 行内公式必须完美
                'fix_display_math': True,  # 行间公式必须完美
                'fix_references': True,  # reference必须完美
                'fix_tables': True,  # 表格必须完美
                'fix_mojibake': True,  # 正文必须完美（修复乱码）
                'fix_headings': True,  # 标题结构必须完美
                'use_crossref': True,  # 使用Crossref增强reference质量
                'use_table_screenshot': True,  # 使用截图确保表格质量
                'batch_size': 50,  # Larger batches
                'timeout_per_task': 300,  # 5 minutes
                'target_time_per_page': None,  # No limit
            },
            'balanced': {
                # 平衡模式：至少标题结构、公式、图片、reference要好
                # Keep page extraction fast; reserve LLM for targeted final cleanup only.
                'use_llm_for_all': False,
                'max_parallel_llm_tasks': 16,  # Higher parallelism for faster post-processing
                'max_parallel_pages': min(20, cpu_count * 3),  # More aggressive parallelization
                'fix_inline_formulas': True,  # 行内公式要好
                'fix_display_math': False,  # Skip heavy display-math cleanup for speed
                'fix_references': False,  # Skip expensive references LLM pass in balanced
                'fix_tables': False,  # 表格可以跳过以节省时间
                'fix_mojibake': True,  # 修复乱码
                'fix_headings': True,  # 标题结构要好
                'use_crossref': False,  # 跳过Crossref以节省时间
                'use_table_screenshot': False,  # 跳过截图以节省时间
                'batch_size': 50,  # Larger batches for efficiency
                'timeout_per_task': 35,  # Hard cap slow cleanup tasks
                'target_time_per_page': 2.3,  # target ~30s for ~13 pages
                'enable_final_llm_cleanup': True,
            },
            'fast': {
                # 快速模式：标题结构、公式要尽可能好
                'use_llm_for_all': False,
                'max_parallel_llm_tasks': 2,
                'max_parallel_pages': min(12, cpu_count * 2),
                'fix_inline_formulas': False,  # Skip final cleanup to hit strict latency target
                'fix_display_math': False,
                'fix_references': False,  # 可以跳过reference
                'fix_tables': False,  # 可以跳过表格
                'fix_mojibake': False,
                'fix_headings': False,
                'use_crossref': False,  # Skip Crossref
                'use_table_screenshot': False,
                'batch_size': 50,
                'timeout_per_task': 20,
                'target_time_per_page': 1.1,  # target ~15s for ~13 pages
                'enable_final_llm_cleanup': False,
            },
            'ultra_fast': {
                # 极速模式：存AI方便读的，当作中间文件就可以
                'use_llm_for_all': False,  # Minimal LLM
                'max_parallel_llm_tasks': 2,
                'max_parallel_pages': min(6, cpu_count),
                'fix_inline_formulas': False,  # 跳过公式修复（中间文件，AI能读即可）
                'fix_display_math': False,  # 跳过公式修复
                'fix_references': False,  # 跳过reference修复
                'fix_tables': False,  # 跳过表格修复
                'fix_mojibake': True,  # 只修复乱码（基本可读性）
                'fix_headings': True,  # 保留标题修复（基本结构）
                'use_crossref': False,
                'use_table_screenshot': False,
                'batch_size': 10,
                'timeout_per_task': 30,  # 30 seconds
                'target_time_per_page': 0.5,  # ~5s for 10 pages
                'enable_final_llm_cleanup': False,
            }
        }
        
        return configs.get(speed_mode, configs['balanced'])
