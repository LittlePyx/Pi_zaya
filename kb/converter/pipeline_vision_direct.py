from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional

try:
    import fitz
except ImportError:
    fitz = None

from .geometry_utils import _rect_area, _bbox_width
from .layout_analysis import _collect_visual_rects
from .heuristics import _page_has_references_heading, _page_looks_like_references_content


def process_batch_vision_direct(self, doc, pdf_path: Path, assets_dir: Path, speed_mode: str = 'normal') -> List[Optional[str]]:
    """
    Bypass all text-extraction / block-classification logic.
    For every page: render a high-DPI screenshot, send it to the vision LLM,
    and collect the Markdown it returns.
    Supports parallel processing with ThreadPoolExecutor.
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

    # Get speed mode config
    import multiprocessing
    speed_config = self._get_speed_mode_config(speed_mode, total_pages)
    cpu_count = multiprocessing.cpu_count()
    
    # DPI from config or environment variable
    # Higher DPI for better formula recognition quality
    base_dpi = int(getattr(self, "dpi", 200) or 200)
    try:
        vision_dpi = int(os.environ.get("KB_PDF_VISION_DPI", "") or "")
        if vision_dpi > 0:
            dpi = max(200, min(600, vision_dpi))  # Minimum 200 for better quality
        else:
            dpi = speed_config.get('dpi', 220)  # Increased from 160 to 220 for better formula recognition
    except Exception:
        dpi = speed_config.get('dpi', 220)  # Increased from 160 to 220
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    # Determine number of workers for parallel processing
    max_parallel = speed_config.get('max_parallel_pages', min(8, max(1, cpu_count)))
    try:
        max_parallel = int(max_parallel)
    except Exception:
        max_parallel = min(8, max(1, cpu_count))
    max_parallel = max(1, min(64, max_parallel, total_pages))

    raw_llm_pw = (os.environ.get("KB_PDF_LLM_PAGE_WORKERS") or "").strip()
    num_workers = int(raw_llm_pw) if raw_llm_pw else int(os.environ.get("KB_PDF_WORKERS", "0") or "0")
    if num_workers <= 0:
        if total_pages <= 2:
            num_workers = 1
        else:
            num_workers = min(max_parallel, cpu_count, total_pages)

    # Effective LLM inflight limit (semaphore slots).
    # Prefer env override; otherwise align with LLMWorker's runtime semaphore limit.
    inflight_source = "speed_config"
    try:
        raw_inflight = (os.environ.get("KB_LLM_MAX_INFLIGHT") or "").strip()
        if raw_inflight:
            max_inflight = int(raw_inflight)
            inflight_source = "env:KB_LLM_MAX_INFLIGHT"
        else:
            worker_inflight = 0
            try:
                worker_inflight = int(self.llm_worker.get_llm_max_inflight())
            except Exception:
                try:
                    worker_inflight = int(getattr(self.llm_worker, "_llm_max_inflight", 0) or 0)
                except Exception:
                    worker_inflight = 0
            if worker_inflight > 0:
                max_inflight = worker_inflight
                inflight_source = "llm_worker"
            else:
                max_inflight = int(speed_config.get('max_inflight', 8) or 8)
        max_inflight = max(1, min(32, int(max_inflight)))
    except Exception:
        max_inflight = max(1, min(32, int(speed_config.get('max_inflight', 8) or 8)))
        inflight_source = "fallback"
    
    num_workers_before_cap = num_workers
    cap = None
    
    # If user explicitly set KB_PDF_LLM_PAGE_WORKERS, don't cap at all (trust the user)
    if raw_llm_pw:
        cap = None
        num_workers = min(int(num_workers), int(total_pages))
    else:
        # Only cap if user didn't explicitly set KB_PDF_LLM_PAGE_WORKERS
        try:
            raw_cap = (os.environ.get("KB_PDF_LLM_PAGE_WORKERS_CAP") or "").strip()
            if raw_cap:
                cap = int(raw_cap)
                cap = max(1, min(64, int(cap)))
            else:
                # Keep page-workers <= effective inflight to avoid semaphore saturation timeouts.
                cap = max(1, min(int(max_parallel), int(max_inflight)))
            
            if cap is not None:
                num_workers = min(int(num_workers), int(cap), int(total_pages))
            else:
                num_workers = min(int(num_workers), int(total_pages))
        except Exception as e:
            # On error, still keep a conservative cap.
            cap = max(1, min(int(max_parallel), int(max_inflight)))
            num_workers = min(int(num_workers), int(cap), int(total_pages))

    # Debug: print worker count calculation
    try:
        print(
            f"[VISION_DIRECT] worker calculation: raw_llm_pw={raw_llm_pw!r}, "
            f"num_workers_before={num_workers_before_cap}, final_num_workers={num_workers}, "
            f"max_inflight={max_inflight} ({inflight_source}), cap={cap}, total_pages={total_pages}",
            flush=True,
        )
        # Warn if max_inflight is less than num_workers (may cause timeouts)
        if max_inflight < num_workers:
            print(f"[VISION_DIRECT] WARNING: KB_LLM_MAX_INFLIGHT={max_inflight} < num_workers={num_workers}. "
                  f"This may cause timeout errors. Consider setting KB_LLM_MAX_INFLIGHT >= {num_workers}", flush=True)
    except Exception:
        pass

    if num_workers <= 1 or total_pages <= 1:
        # Sequential processing
        print(f"[VISION_DIRECT] Converting pages {start+1}-{end} via VL screenshots (dpi={dpi}, sequential)", flush=True)
        for i in range(start, end):
            t0 = time.time()
            print(f"Processing page {i+1}/{total_pages} (vision-direct) ...", flush=True)
            try:
                page = doc.load_page(i)
                page_w = float(page.rect.width)
                page_h = float(page.rect.height)

                is_references_page = False
                try:
                    is_references_page = bool(
                        _page_has_references_heading(page) or _page_looks_like_references_content(page)
                    )
                except Exception:
                    is_references_page = False
                metadata_rects: list[fitz.Rect] = []
                if not is_references_page:
                    try:
                        metadata_rects = self._collect_non_body_metadata_rects(
                            page,
                            page_index=i,
                            is_references_page=False,
                        )
                    except Exception:
                        metadata_rects = []
                
                # Extract images BEFORE sending to LLM, so they're available when LLM references them
                image_names: list[str] = []
                figure_entries: list[dict] = []
                figure_meta_by_asset: dict[str, dict] = {}
                visual_rects: list[fitz.Rect] = []
                try:
                    visual_rects = _collect_visual_rects(page)
                    W = page_w
                    H = page_h
                    header_threshold = H * 0.12
                    footer_threshold = H * 0.88
                    side_margin = W * 0.05
                    spanning_threshold = W * 0.55
                    line_boxes = self._collect_page_text_line_boxes(page)
                    
                    img_count = 0
                    for rect_idx, rect in enumerate(visual_rects):
                        # Check if this is a full-width image
                        is_full_width = _rect_area(rect) >= (W * H * 0.40) or _bbox_width(tuple(rect)) >= spanning_threshold
                        
                        # Skip if in header/footer region (unless it's a large figure)
                        if rect.y1 < header_threshold or rect.y0 > footer_threshold:
                            if not is_full_width and _rect_area(rect) < (W * H * 0.15):
                                continue
                        
                        # Skip small edge artifacts
                        if not is_full_width:
                            if rect.x0 < side_margin and rect.width < W * 0.1:
                                continue
                            if rect.x1 > W - side_margin and rect.width < W * 0.1:
                                continue
                        
                        cropped_rect = self._expanded_visual_crop_rect(
                            rect=rect,
                            page_w=W,
                            page_h=H,
                            is_full_width=bool(is_full_width),
                            line_boxes=line_boxes,
                        )
                        
                        if cropped_rect.width <= 0 or cropped_rect.height <= 0:
                            continue
                        
                        # Save image
                        img_name = f"page_{i+1}_fig_{rect_idx+1}.png"
                        img_path = assets_dir / img_name
                        used_clip = cropped_rect
                        
                        try:
                            pix_img = page.get_pixmap(clip=cropped_rect, dpi=dpi)
                            pix_img.save(img_path)
                            # Verify file was saved
                            if img_path.exists() and img_path.stat().st_size >= 256:
                                img_count += 1
                                if img_name not in image_names:
                                    image_names.append(img_name)
                                figure_entries.append(
                                    {
                                        "asset_name": img_name,
                                        "bbox": [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)],
                                        "crop_bbox": [float(used_clip.x0), float(used_clip.y0), float(used_clip.x1), float(used_clip.y1)],
                                    }
                                )
                        except Exception:
                            # Fallback to original rect if crop fails
                            try:
                                used_clip = rect
                                pix_img = page.get_pixmap(clip=rect, dpi=dpi)
                                pix_img.save(img_path)
                                if img_path.exists() and img_path.stat().st_size >= 256:
                                    img_count += 1
                                    if img_name not in image_names:
                                        image_names.append(img_name)
                                    figure_entries.append(
                                        {
                                            "asset_name": img_name,
                                            "bbox": [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)],
                                            "crop_bbox": [float(used_clip.x0), float(used_clip.y0), float(used_clip.x1), float(used_clip.y1)],
                                        }
                                    )
                            except Exception:
                                continue
                    
                    if img_count > 0:
                        print(f"  [Page {i+1}] Extracted {img_count} images", flush=True)
                except Exception as e:
                    print(f"  [Page {i+1}] Image extraction failed: {e}", flush=True)
                    # Continue even if image extraction fails

                try:
                    if figure_entries:
                        cap_candidates = self._extract_page_figure_caption_candidates(page)
                        figure_entries = self._match_figure_entries_with_captions(
                            page=page,
                            figure_entries=figure_entries,
                            caption_candidates=cap_candidates,
                        )
                    figure_meta_by_asset = self._persist_page_figure_metadata(
                        assets_dir=assets_dir,
                        page_index=i,
                        figure_entries=figure_entries,
                    )
                except Exception as e:
                    print(f"  [Page {i+1}] Figure metadata persist skipped: {e}", flush=True)
                    figure_meta_by_asset = {}
                
                pix = page.get_pixmap(matrix=mat, alpha=False)
                png_bytes = pix.tobytes("png")
                
                # Compress PNG based on speed mode config
                try:
                    compress_level_raw = os.environ.get("KB_PDF_VISION_COMPRESS", "").strip()
                    if compress_level_raw:
                        compress_level = int(compress_level_raw)
                    else:
                        compress_level = speed_config.get('compress', 3)
                    if compress_level > 0:
                        try:
                            from PIL import Image
                            import io
                            img = Image.open(io.BytesIO(png_bytes))
                            output = io.BytesIO()
                            img.save(output, format="PNG", optimize=True, compress_level=min(9, max(1, compress_level)))
                            png_bytes = output.getvalue()
                        except ImportError:
                            pass  # PIL not available, skip compression
                except Exception:
                    pass

                if metadata_rects:
                    png_bytes = self._mask_rects_on_png(
                        png_bytes,
                        rects=metadata_rects,
                        page_width=page_w,
                        page_height=page_h,
                    )
                    try:
                        print(f"  [Page {i+1}] Masked {len(metadata_rects)} metadata region(s) before VL OCR", flush=True)
                    except Exception:
                        pass

                # Save screenshot for debugging if requested
                try:
                    if bool(int(os.environ.get("KB_PDF_SAVE_PAGE_SCREENSHOTS", "0") or "0")):
                        dbg_dir = assets_dir / "page_screenshots"
                        dbg_dir.mkdir(exist_ok=True)
                        (dbg_dir / f"page_{i+1}.png").write_bytes(png_bytes)
                except Exception:
                    pass

                page_hint = ""
                if is_references_page:
                    page_hint = (
                        "This page is likely in the REFERENCES/BIBLIOGRAPHY section. "
                        "Output plain-text references only, one complete reference per line. "
                        "Do not use $...$ or $$...$$ on this page."
                    )
                else:
                    if i == 0:
                        page_hint = (
                            "This is the first page of the paper. "
                            "Preserve the Abstract from its very first visible sentence, in normal reading order. "
                            "Do not drop the abstract opening because of nearby title, author, or figure content."
                        )
                if (not is_references_page) and image_names:
                    show_n = 6
                    paths = ", ".join(f"./assets/{nm}" for nm in image_names[:show_n])
                    hint_img = (
                        "This page has extracted figure images available in assets. "
                        f"If figures appear on this page, embed them with exact Markdown links using these paths: {paths}."
                    )
                    page_hint = (page_hint + " " + hint_img).strip() if page_hint else hint_img
                    hint_map = self._build_page_figure_mapping_hint(figure_meta_by_asset)
                    if hint_map:
                        page_hint = (page_hint + " " + hint_map).strip()
                formula_placeholders: dict[str, str] = {}
                if (not is_references_page) and self._vision_formula_overlay_enabled():
                    try:
                        eq_candidates = self._collect_display_math_candidates(
                            page,
                            page_index=i,
                            is_references_page=False,
                        )
                        if eq_candidates:
                            eq_labeled_rects, formula_placeholders = self._extract_formula_placeholders_for_page(
                                page,
                                page_index=i,
                                candidates=eq_candidates,
                                base_dpi=int(dpi),
                            )
                            if eq_labeled_rects and formula_placeholders:
                                png_bytes = self._mask_rects_with_labels_on_png(
                                    png_bytes,
                                    labeled_rects=eq_labeled_rects,
                                    page_width=page_w,
                                    page_height=page_h,
                                )
                                hint_eq = self._formula_placeholder_hint(formula_placeholders)
                                if hint_eq:
                                    page_hint = (page_hint + " " + hint_eq).strip() if page_hint else hint_eq
                                print(
                                    f"  [Page {i+1}] Formula overlay enabled: {len(formula_placeholders)} display equation(s)",
                                    flush=True,
                                )
                    except Exception as e:
                        print(f"  [Page {i+1}] Formula overlay skipped: {e}", flush=True)

                md = self._convert_page_with_vision_guardrails(
                    png_bytes=png_bytes,
                    page=page,
                    page_index=i,
                    total_pages=total_pages,
                    page_hint=page_hint,
                    speed_mode=speed_mode,
                    is_references_page=is_references_page,
                    pdf_path=pdf_path,
                    assets_dir=assets_dir,
                    image_names=image_names,
                    formula_placeholders=formula_placeholders,
                )
                elapsed = time.time() - t0
                if md:
                    md = self._postprocess_vision_page_markdown(
                        md,
                        page=page,
                        page_index=i,
                        pdf_path=pdf_path,
                        assets_dir=assets_dir,
                        image_names=image_names,
                        figure_meta_by_asset=figure_meta_by_asset,
                        visual_rects=visual_rects,
                        is_references_page=is_references_page,
                    )
                    results[i] = md
                    print(f"Finished page {i+1}/{total_pages} ({elapsed:.1f}s, {len(md)} chars)", flush=True)
                else:
                    print(f"[VISION_DIRECT] page {i+1} failed in both VL and fallback paths", flush=True)
            except Exception as e:
                error_str = str(e)
                print(f"[VISION_DIRECT] error page {i+1}: {e}", flush=True)
                
                # Check for critical API errors that should stop processing
                if "Access denied" in error_str or "account is in good standing" in error_str:
                    print("[VISION_DIRECT] API access denied detected; check account status.", flush=True)
                    # Continue processing other pages, but mark this one as failed
                
                import traceback
                traceback.print_exc()
                results[i] = None
        return results

    # Parallel processing
    print(f"[VISION_DIRECT] Converting pages {start+1}-{end} via VL screenshots (dpi={dpi}, {num_workers} workers)", flush=True)
    from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

    def process_single_page(i: int):
        try:
            print(f"Processing page {i+1}/{total_pages} (vision-direct) ...", flush=True)
            t0 = time.time()
            # Avoid sharing fitz.Document / Page across threads
            with fitz.open(str(pdf_path)) as local_doc:
                page = local_doc.load_page(i)
                page_w = float(page.rect.width)
                page_h = float(page.rect.height)

                is_references_page = False
                try:
                    is_references_page = bool(
                        _page_has_references_heading(page) or _page_looks_like_references_content(page)
                    )
                except Exception:
                    is_references_page = False
                metadata_rects: list[fitz.Rect] = []
                if not is_references_page:
                    try:
                        metadata_rects = self._collect_non_body_metadata_rects(
                            page,
                            page_index=i,
                            is_references_page=False,
                        )
                    except Exception:
                        metadata_rects = []
                
                # Extract images BEFORE sending to LLM, so they're available when LLM references them
                image_names: list[str] = []
                figure_entries: list[dict] = []
                figure_meta_by_asset: dict[str, dict] = {}
                visual_rects: list[fitz.Rect] = []
                try:
                    visual_rects = _collect_visual_rects(page)
                    W = page_w
                    H = page_h
                    header_threshold = H * 0.12
                    footer_threshold = H * 0.88
                    side_margin = W * 0.05
                    spanning_threshold = W * 0.55
                    line_boxes = self._collect_page_text_line_boxes(page)
                    
                    img_count = 0
                    for rect_idx, rect in enumerate(visual_rects):
                        # Check if this is a full-width image
                        is_full_width = _rect_area(rect) >= (W * H * 0.40) or _bbox_width(tuple(rect)) >= spanning_threshold
                        
                        # Skip if in header/footer region (unless it's a large figure)
                        if rect.y1 < header_threshold or rect.y0 > footer_threshold:
                            if not is_full_width and _rect_area(rect) < (W * H * 0.15):
                                continue
                        
                        # Skip small edge artifacts
                        if not is_full_width:
                            if rect.x0 < side_margin and rect.width < W * 0.1:
                                continue
                            if rect.x1 > W - side_margin and rect.width < W * 0.1:
                                continue
                        
                        cropped_rect = self._expanded_visual_crop_rect(
                            rect=rect,
                            page_w=W,
                            page_h=H,
                            is_full_width=bool(is_full_width),
                            line_boxes=line_boxes,
                        )
                        
                        if cropped_rect.width <= 0 or cropped_rect.height <= 0:
                            continue
                        
                        # Save image
                        img_name = f"page_{i+1}_fig_{rect_idx+1}.png"
                        img_path = assets_dir / img_name
                        used_clip = cropped_rect
                        
                        try:
                            pix_img = page.get_pixmap(clip=cropped_rect, dpi=dpi)
                            pix_img.save(img_path)
                            # Verify file was saved
                            if img_path.exists() and img_path.stat().st_size >= 256:
                                img_count += 1
                                if img_name not in image_names:
                                    image_names.append(img_name)
                                figure_entries.append(
                                    {
                                        "asset_name": img_name,
                                        "bbox": [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)],
                                        "crop_bbox": [float(used_clip.x0), float(used_clip.y0), float(used_clip.x1), float(used_clip.y1)],
                                    }
                                )
                        except Exception:
                            # Fallback to original rect if crop fails
                            try:
                                used_clip = rect
                                pix_img = page.get_pixmap(clip=rect, dpi=dpi)
                                pix_img.save(img_path)
                                if img_path.exists() and img_path.stat().st_size >= 256:
                                    img_count += 1
                                    if img_name not in image_names:
                                        image_names.append(img_name)
                                    figure_entries.append(
                                        {
                                            "asset_name": img_name,
                                            "bbox": [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)],
                                            "crop_bbox": [float(used_clip.x0), float(used_clip.y0), float(used_clip.x1), float(used_clip.y1)],
                                        }
                                    )
                            except Exception:
                                continue
                    
                    if img_count > 0:
                        print(f"  [Page {i+1}] Extracted {img_count} images", flush=True)
                except Exception as e:
                    print(f"  [Page {i+1}] Image extraction failed: {e}", flush=True)
                    # Continue even if image extraction fails

                try:
                    if figure_entries:
                        cap_candidates = self._extract_page_figure_caption_candidates(page)
                        figure_entries = self._match_figure_entries_with_captions(
                            page=page,
                            figure_entries=figure_entries,
                            caption_candidates=cap_candidates,
                        )
                    figure_meta_by_asset = self._persist_page_figure_metadata(
                        assets_dir=assets_dir,
                        page_index=i,
                        figure_entries=figure_entries,
                    )
                except Exception as e:
                    print(f"  [Page {i+1}] Figure metadata persist skipped: {e}", flush=True)
                    figure_meta_by_asset = {}
                
                pix = page.get_pixmap(matrix=mat, alpha=False)
                png_bytes = pix.tobytes("png")
                
                # Compress PNG based on speed mode config
                try:
                    compress_level_raw = os.environ.get("KB_PDF_VISION_COMPRESS", "").strip()
                    if compress_level_raw:
                        compress_level = int(compress_level_raw)
                    else:
                        compress_level = speed_config.get('compress', 3)
                    if compress_level > 0:
                        # Use PIL/Pillow to compress if available
                        try:
                            from PIL import Image
                            import io
                            img = Image.open(io.BytesIO(png_bytes))
                            output = io.BytesIO()
                            # compress_level: 1-9, higher = more compression (slower)
                            img.save(output, format="PNG", optimize=True, compress_level=min(9, max(1, compress_level)))
                            png_bytes = output.getvalue()
                        except ImportError:
                            pass  # PIL not available, skip compression
                except Exception:
                    pass

                if metadata_rects:
                    png_bytes = self._mask_rects_on_png(
                        png_bytes,
                        rects=metadata_rects,
                        page_width=page_w,
                        page_height=page_h,
                    )
                    try:
                        print(f"  [Page {i+1}] Masked {len(metadata_rects)} metadata region(s) before VL OCR", flush=True)
                    except Exception:
                        pass

                # Save screenshot for debugging if requested
                try:
                    if bool(int(os.environ.get("KB_PDF_SAVE_PAGE_SCREENSHOTS", "0") or "0")):
                        dbg_dir = assets_dir / "page_screenshots"
                        dbg_dir.mkdir(exist_ok=True)
                        (dbg_dir / f"page_{i+1}.png").write_bytes(png_bytes)
                except Exception:
                    pass

                page_hint = ""
                if is_references_page:
                    page_hint = (
                        "This page is likely in the REFERENCES/BIBLIOGRAPHY section. "
                        "Output plain-text references only, one complete reference per line. "
                        "Do not use $...$ or $$...$$ on this page."
                    )
                else:
                    if i == 0:
                        page_hint = (
                            "This is the first page of the paper. "
                            "Preserve the Abstract from its very first visible sentence, in normal reading order. "
                            "Do not drop the abstract opening because of nearby title, author, or figure content."
                        )
                if (not is_references_page) and image_names:
                    show_n = 6
                    paths = ", ".join(f"./assets/{nm}" for nm in image_names[:show_n])
                    hint_img = (
                        "This page has extracted figure images available in assets. "
                        f"If figures appear on this page, embed them with exact Markdown links using these paths: {paths}."
                    )
                    page_hint = (page_hint + " " + hint_img).strip() if page_hint else hint_img
                    hint_map = self._build_page_figure_mapping_hint(figure_meta_by_asset)
                    if hint_map:
                        page_hint = (page_hint + " " + hint_map).strip()
                formula_placeholders: dict[str, str] = {}
                if (not is_references_page) and self._vision_formula_overlay_enabled():
                    try:
                        eq_candidates = self._collect_display_math_candidates(
                            page,
                            page_index=i,
                            is_references_page=False,
                        )
                        if eq_candidates:
                            eq_labeled_rects, formula_placeholders = self._extract_formula_placeholders_for_page(
                                page,
                                page_index=i,
                                candidates=eq_candidates,
                                base_dpi=int(dpi),
                            )
                            if eq_labeled_rects and formula_placeholders:
                                png_bytes = self._mask_rects_with_labels_on_png(
                                    png_bytes,
                                    labeled_rects=eq_labeled_rects,
                                    page_width=page_w,
                                    page_height=page_h,
                                )
                                hint_eq = self._formula_placeholder_hint(formula_placeholders)
                                if hint_eq:
                                    page_hint = (page_hint + " " + hint_eq).strip() if page_hint else hint_eq
                                print(
                                    f"  [Page {i+1}] Formula overlay enabled: {len(formula_placeholders)} display equation(s)",
                                    flush=True,
                                )
                    except Exception as e:
                        print(f"  [Page {i+1}] Formula overlay skipped: {e}", flush=True)

                md = self._convert_page_with_vision_guardrails(
                    png_bytes=png_bytes,
                    page=page,
                    page_index=i,
                    total_pages=total_pages,
                    page_hint=page_hint,
                    speed_mode=speed_mode,
                    is_references_page=is_references_page,
                    pdf_path=pdf_path,
                    assets_dir=assets_dir,
                    image_names=image_names,
                    formula_placeholders=formula_placeholders,
                )
                elapsed = time.time() - t0
                if md:
                    md = self._postprocess_vision_page_markdown(
                        md,
                        page=page,
                        page_index=i,
                        pdf_path=pdf_path,
                        assets_dir=assets_dir,
                        image_names=image_names,
                        figure_meta_by_asset=figure_meta_by_asset,
                        visual_rects=visual_rects,
                        is_references_page=is_references_page,
                    )
                    print(f"Finished page {i+1}/{total_pages} ({elapsed:.1f}s, {len(md)} chars)", flush=True)
                    return i, md
                else:
                    print(f"[VISION_DIRECT] page {i+1} failed in both VL and fallback paths", flush=True)
                    return i, None
        except Exception as e:
            error_str = str(e)
            print(f"[VISION_DIRECT] error page {i+1}: {e}", flush=True)
            
            # Check for critical API errors
            if "Access denied" in error_str or "account is in good standing" in error_str:
                print("[VISION_DIRECT] API access denied detected; check account status.", flush=True)
            
            import traceback
            traceback.print_exc()
            return i, None

    executor = ThreadPoolExecutor(max_workers=num_workers)
    futures = {executor.submit(process_single_page, i): i for i in range(start, end)}
    pending = set(futures.keys())
    done_pages = set()

    # Heartbeat logging to avoid UI appearing frozen
    hb_every_s = 8.0
    try:
        hb_every_s = float(os.environ.get("KB_PDF_BATCH_HEARTBEAT_S", str(hb_every_s)) or hb_every_s)
        hb_every_s = max(2.0, min(60.0, hb_every_s))
    except Exception:
        hb_every_s = 8.0
    last_hb = time.time()

    try:
        while pending:
            now_ts = time.time()
            done, not_done = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
            pending = set(not_done)
            try:
                if (now_ts - last_hb) >= hb_every_s:
                    inflight_pages = sorted({int(futures[fut]) + 1 for fut in pending})
                    if inflight_pages:
                        head = inflight_pages[:12]
                        more = len(inflight_pages) - len(head)
                        extra = f" (+{more} more)" if more > 0 else ""
                        print(
                            f"[VISION_DIRECT] still running pages: {head}{extra} | workers={num_workers} llm_inflight={max_inflight}",
                            flush=True,
                        )
                    last_hb = now_ts
            except Exception:
                pass
            for future in done:
                i = futures[future]
                try:
                    i2, result = future.result()
                    results[i2] = result
                    done_pages.add(i2 + 1)
                except Exception as e:
                    print(f"[VISION_DIRECT] error processing page {i+1}: {e}", flush=True)
                    results[i] = None
    finally:
        if pending:
            for future in pending:
                i = futures.get(future)
                if i is not None and results[i] is None:
                    results[i] = f"<!-- kb_page: {i+1} -->\n\n[Page {i+1} conversion incomplete]"
                try:
                    future.cancel()
                except Exception:
                    pass
        executor.shutdown(wait=False, cancel_futures=True)

    return results


