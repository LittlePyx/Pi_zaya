from __future__ import annotations

import re
import time
from pathlib import Path
from typing import List, Tuple

try:
    import fitz
except ImportError:
    fitz = None

from .block_classifier import _looks_like_code_block, _looks_like_math_block
from .geometry_utils import _bbox_width, _rect_area, _rect_intersection_area, _union_rect
from .heuristics import (
    _is_caption_like_text,
    _is_noise_line,
    _is_non_body_metadata_text,
    _looks_like_equation_text,
    detect_header_tag,
)
from .layout_analysis import _detect_column_split_x, _is_frontmatter_noise_line, sort_blocks_reading_order
from .models import TextBlock
from .text_utils import _normalize_text


def extract_text_blocks(
    converter,
    page,
    page_index: int,
    body_size: float,
    tables: List[Tuple["fitz.Rect", str]],
    visual_rects: List["fitz.Rect"],
    assets_dir: Path,
    is_references_page: bool = False,
) -> List[TextBlock]:
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

        line_items: list[tuple["fitz.Rect", str]] = []

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

        def _looks_like_running_header_footer_line(line_rect: "fitz.Rect", line_text: str) -> bool:
            norm = _normalize_text(line_text).strip()
            if not norm:
                return False
            if not (float(line_rect.y1) <= H * 0.09 or float(line_rect.y0) >= H * 0.94):
                return False
            has_page_marker = bool(re.search(r"\bpage\s+\d+\s+of\s+\d+\b", norm, flags=re.IGNORECASE))
            has_journal_marker = bool(
                re.search(
                    r"\b(?:light:\s*science\s*&\s*applications|optics express|nature photonics|natural photonics|advanced science news)\b",
                    norm,
                    flags=re.IGNORECASE,
                )
            )
            has_volume_marker = bool(re.search(r"\(\d{4}\)\s+\d+[:\s]\d+", norm))
            if has_page_marker and (has_journal_marker or has_volume_marker):
                return True
            if float(line_rect.width) < W * 0.55:
                return False
            if has_page_marker:
                return True
            if has_journal_marker:
                return True
            return False

        def _looks_like_footer_page_number_line(line_rect: "fitz.Rect", line_text: str) -> bool:
            norm = _normalize_text(line_text).strip()
            if not re.fullmatch(r"\d{1,4}", norm):
                return False
            cy = (float(line_rect.y0) + float(line_rect.y1)) * 0.5
            if cy < H * 0.92:
                return False
            cx = (float(line_rect.x0) + float(line_rect.x1)) * 0.5
            if abs(cx - (W * 0.5)) > W * 0.12:
                return False
            return float(line_rect.width) <= W * 0.10

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
            if _looks_like_running_header_footer_line(lb, line_text):
                continue
            if _looks_like_footer_page_number_line(lb, line_text):
                continue
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
            # Numbered section headings should not be treated as math even if OCR
            # heuristics are tempted by the leading numeral.
            if re.match(r"^\d+(?:\.\d+)*\.?\s+[A-Z][A-Za-z][A-Za-z0-9\-\s]{1,80}$", tt):
                return False
            # Citation-rich prose lines are common in papers and should not be wrapped
            # into display math just because they contain brackets and a few symbols.
            try:
                cite_n = len(re.findall(r"\[\s*\d{1,4}(?:\s*[,;\-\u2013]\s*\d{1,4})*\s*\]", tt))
                word_n = len(re.findall(r"\b[A-Za-z]{2,}\b", tt))
                hard_math_n = len(re.findall(r"(?:=|\\sum|\\int|\\frac|\\sqrt|\^|_|[\u2208\u2264\u2265\u2248\u00D7\u00B7\u03A3\u2212\u222B])", tt))
                if word_n >= 6 and (cite_n >= 1 or tt.endswith((".", ":", ";", ","))) and hard_math_n <= 1:
                    return False
                if word_n >= 8 and hard_math_n == 0:
                    return False
            except Exception:
                pass
            if _looks_like_math_block([tt]):
                return True
            if _looks_like_equation_text(tt):
                return True
            # Extra: short symbol-heavy lines are usually equation lines
            sym_n = len(re.findall(r"[=+\-*/^_{}\\\[\]]|[\u2208\u2264\u2265\u2248\u00D7\u00B7\u03A3\u2212\u222B]", tt))
            word_n = len(re.findall(r"\b[A-Za-z]{2,}\b", tt))
            if len(tt) <= 80 and sym_n >= 3 and word_n <= 10:
                return True
            return False

        groups: list[tuple[bool, list[tuple["fitz.Rect", str]]]] = []
        cur_is_math = _is_math_line(line_items[0][1])
        cur: list[tuple["fitz.Rect", str]] = []
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

            def _looks_like_figure_internal_label() -> bool:
                if pre_is_caption or (not visual_rects):
                    return False
                norm = _normalize_text(full_text_space).strip()
                if not norm:
                    return False
                rect = fitz.Rect(group_bbox)
                cx = (float(rect.x0) + float(rect.x1)) / 2.0
                cy = (float(rect.y0) + float(rect.y1)) / 2.0
                pad = max(18.0, min(float(W), float(H)) * 0.022)
                near_visual = False
                for vr in visual_rects:
                    try:
                        ex = fitz.Rect(
                            float(vr.x0) - pad,
                            float(vr.y0) - pad,
                            float(vr.x1) + pad,
                            float(vr.y1) + pad,
                        )
                    except Exception:
                        continue
                    if (float(ex.x0) <= cx <= float(ex.x1)) and (float(ex.y0) <= cy <= float(ex.y1)):
                        near_visual = True
                        break
                    try:
                        if _rect_intersection_area(rect, ex) > 0.0:
                            near_visual = True
                            break
                    except Exception:
                        pass
                if not near_visual:
                    return False

                words = re.findall(r"[A-Za-z]{1,12}", norm)
                if len(norm) <= 24 and re.fullmatch(r"(?:[a-z]|[A-Z]{1,3})(?:\s*/\s*(?:[a-z]|[A-Z]{1,3}))*", norm):
                    return True
                if len(norm) <= 64 and len(words) <= 8 and (not re.search(r"[.!?;:]\s*$", norm)):
                    return True
                if (
                    len(norm) <= 72
                    and len(words) <= 10
                    and float(rect.height) <= max(18.0, float(H) * 0.04)
                    and float(max_size) <= float(body_size) + 1.6
                ):
                    return True
                return False

            # Source-level suppression: strip author/affiliation/journal metadata blocks
            # before they enter body reconstruction.
            if not pre_is_caption:
                if _looks_like_figure_internal_label():
                    continue
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
            if re.search(r"^\s*[A-Z][a-z]+\s+\d+\s*,.*\d+\s*,.*and.*\d+\s*,?\s*$", full_text_space):
                tb = TextBlock(
                    bbox=tuple(group_bbox),
                    text=full_text_space,
                    max_font_size=max_size,
                    is_bold=is_bold,
                    heading_level=None,
                    is_math=False,
                    is_code=False,
                    is_caption=False,
                )
                text_blocks.append(tb)
                continue

            # Detect math formulas, code blocks, and captions early
            is_math = False
            is_code = False
            is_caption = False

            # Check if this is a caption (Figure/Table caption)
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
                    if full_text_space in converter.noise_texts or _is_noise_line(full_text_space):
                        continue

            if full_text_space in converter.noise_texts:
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
                    looks_like_numbered_heading = bool(
                        re.match(r"^\d+(?:\.\d+)*\.?\s+[A-Z][A-Za-z]", text_stripped)
                    )
                    # Check for formula-like patterns more strictly
                    if (not looks_like_numbered_heading) and (
                       re.search(r"^\s*[A-Z]?\s*\d+\s*[a-z]", text_stripped) or \
                       re.search(r"^\s*\d+\s*[a-z]+\s*[+\-]", text_stripped) or \
                       re.search(r"^\s*[a-z]\s*\d+", text_stripped) or \
                       re.fullmatch(r"(?:[A-Za-z]\s+){1,4}[A-Za-z]", text_stripped) or \
                       (re.search(r"\d+.*[a-z]|[a-z].*\d+", text_stripped) and not re.search(r"[A-Z]{2,}", text_stripped) and "=" not in text_stripped)
                    ):
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
                is_caption=is_caption,
            )
            text_blocks.append(tb)

    # Insert Tables as Blocks
    for rect, md in tables:
        tb = TextBlock(
            bbox=tuple(rect),
            text="[TABLE]",
            max_font_size=body_size,
            is_table=True,
            table_markdown=md,
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
    line_boxes = converter._collect_page_text_line_boxes(page)

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

        cropped_rect = converter._expanded_visual_crop_rect(
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

        def _record_saved_asset(saved_clip: "fitz.Rect") -> None:
            nonlocal img_count
            tb = TextBlock(
                bbox=tuple(rect),
                text=f"![Figure](./assets/{img_name})",
                max_font_size=body_size,
            )
            text_blocks.append(tb)
            figure_entries.append(
                {
                    "asset_name": img_name,
                    "bbox": [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)],
                    "crop_bbox": [float(saved_clip.x0), float(saved_clip.y0), float(saved_clip.x1), float(saved_clip.y1)],
                    "_text_block_index": len(text_blocks) - 1,
                }
            )
            img_count += 1

        try:
            if img_path.exists() and img_path.stat().st_size >= 256:
                _record_saved_asset(used_clip)
                continue
        except Exception:
            pass

        # Use configured DPI
        dpi = converter.dpi
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
                if img_path.exists() and img_path.stat().st_size >= 256:
                    _record_saved_asset(used_clip)
                    continue
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

        _record_saved_asset(used_clip)
        if time.time() - img_step_start > 0.5:
            print(f"      [Page {page_index+1}] Image {rect_idx+1} processing took {time.time()-img_step_start:.2f}s", flush=True)

    print(f"    [Page {page_index+1}] Image processing: {time.time()-step_start:.2f}s, processed {img_count}/{len(visual_rects)} images", flush=True)
    try:
        if figure_entries:
            cap_candidates = converter._extract_page_figure_caption_candidates(page)
            figure_entries = converter._match_figure_entries_with_captions(
                page=page,
                figure_entries=figure_entries,
                caption_candidates=cap_candidates,
            )
            for ent in figure_entries:
                try:
                    block_idx = int(ent.get("_text_block_index"))
                except Exception:
                    block_idx = -1
                if block_idx < 0 or block_idx >= len(text_blocks):
                    continue
                try:
                    fig_no = int(ent.get("fig_no") or 0)
                except Exception:
                    fig_no = 0
                if fig_no <= 0:
                    continue
                asset_name = str(ent.get("asset_name") or "").strip()
                if not asset_name:
                    continue
                text_blocks[block_idx].text = f"![Figure {fig_no}](./assets/{asset_name})"
        converter._persist_page_figure_metadata(
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
