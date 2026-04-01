from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

try:
    import fitz
except ImportError:
    fitz = None

from .geometry_utils import _overlap_1d, _rect_area
from .heuristics import _page_has_references_heading, _page_looks_like_references_content
from .layout_analysis import _collect_visual_rects, detect_body_font_size
from .tables import _extract_tables_by_layout, _page_maybe_has_table_from_dict
from .reference_markdown import normalize_references_page_text


def prepare_page_render_input(
    converter,
    page,
    page_index: int,
    pdf_path: Path,
    assets_dir: Path,
) -> dict:
    page_start = time.time()

    # 1. Analyze Layout
    step_start = time.time()
    body_size = detect_body_font_size([page])  # Heuristic was on full doc, but per-page is okay fallback
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
        cap_candidates = converter._extract_page_figure_caption_candidates(page)
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
        if (wide_banner or tiny_badge) and (not _near_caption(r)):
            continue
        filtered_visual_rects.append(r)
    visual_rects = converter._split_visual_rects_by_internal_captions(
        page=page,
        visual_rects=filtered_visual_rects,
        caption_candidates=cap_candidates,
    )
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
        traceback.print_exc()
        tables_found = []

    # 5. Extract Text Blocks
    step_start = time.time()
    blocks = converter._extract_text_blocks(
        page,
        page_index=page_index,
        body_size=body_size,
        tables=tables_found,
        visual_rects=visual_rects,
        assets_dir=assets_dir,
        is_references_page=is_references_page,
    )
    print(f"  [Page {page_index+1}] Step 5 (text blocks): {time.time()-step_start:.2f}s, found {len(blocks)} blocks", flush=True)

    # 5.5 Merge split math fragments BEFORE any LLM work / rendering.
    # PDF extraction frequently splits a single equation into many tiny blocks ("N X", ", (5)", "r in R").
    # If we try to repair each fragment in isolation, LaTeX quality collapses.
    step_start = time.time()
    try:
        blocks = converter._merge_adjacent_math_fragments(blocks, page_wh=(page.rect.width, page.rect.height))
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
    speed_cfg = getattr(converter, "_active_speed_config", None) or {}
    if converter.cfg.llm and speed_cfg.get("use_llm_for_all", True):
        # Enhance blocks with LLM
        blocks = converter._enhance_blocks_with_llm(blocks, page_index, page)
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
        if converter.cfg.llm and speed_cfg.get("use_llm_for_all", True):
            step_start2 = time.time()
            blocks = converter._merge_adjacent_math_fragments(blocks, page_wh=(page.rect.width, page.rect.height))
            if bool(int(os.environ.get("KB_PDF_DEBUG_MATH", "0") or "0")):
                math_n2 = sum(1 for b in blocks if bool(getattr(b, "is_math", False)))
                print(
                    f"  [Page {page_index+1}] Debug: post-enhance merge math={math_n2} blocks={len(blocks)} ({time.time()-step_start2:.2f}s)",
                    flush=True,
                )
    except Exception:
        pass

    page_no = int(page_index) + 1
    figure_meta_by_asset: dict[str, dict] = {}
    image_names: list[str] = []
    meta_path = assets_dir / f"page_{page_no}_fig_index.json"
    try:
        if meta_path.exists():
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            figures = payload.get("figures") if isinstance(payload, dict) else None
            if isinstance(figures, list):
                for item in figures:
                    if not isinstance(item, dict):
                        continue
                    asset_name = str(item.get("asset_name") or "").strip()
                    if not asset_name:
                        continue
                    figure_meta_by_asset[asset_name] = dict(item)
                image_names = [str(item.get("asset_name") or "").strip() for item in figures if isinstance(item, dict) and str(item.get("asset_name") or "").strip()]
    except Exception:
        figure_meta_by_asset = {}
        image_names = []

    return {
        "blocks": blocks,
        "is_references_page": bool(is_references_page),
        "reference_page_text": (page.get_text("text") if bool(is_references_page) else ""),
        "image_names": image_names,
        "figure_meta_by_asset": figure_meta_by_asset,
        "pdf_path": str(pdf_path),
        "visual_rects": visual_rects,
        "prepare_elapsed": time.time() - page_start,
    }


def render_prepared_page(
    converter,
    *,
    prepared: dict,
    page,
    page_index: int,
    assets_dir: Path,
) -> str:
    step_start = time.time()
    blocks = list(prepared.get("blocks") or [])
    is_references_page = bool(prepared.get("is_references_page", False))
    reference_page_text = str(prepared.get("reference_page_text") or "")
    if is_references_page and reference_page_text.strip():
        result = normalize_references_page_text(reference_page_text)
    else:
        result = converter._render_blocks_to_markdown(
            blocks,
            page_index,
            page=page,
            assets_dir=assets_dir,
            is_references_page=is_references_page,
        )
    image_names = [str(x).strip() for x in list(prepared.get("image_names") or []) if str(x).strip()]
    figure_meta_by_asset = prepared.get("figure_meta_by_asset") if isinstance(prepared.get("figure_meta_by_asset"), dict) else None
    pdf_path_raw = str(prepared.get("pdf_path") or "").strip()
    if result and pdf_path_raw:
        try:
            result = converter._postprocess_vision_page_markdown(
                result,
                page=page,
                page_index=page_index,
                pdf_path=Path(pdf_path_raw),
                assets_dir=assets_dir,
                image_names=image_names,
                figure_meta_by_asset=figure_meta_by_asset,
                visual_rects=list(prepared.get("visual_rects") or []),
                is_references_page=is_references_page,
            )
        except Exception:
            pass
    render_elapsed = time.time() - step_start
    print(f"  [Page {page_index+1}] Step 7 (render): {render_elapsed:.2f}s", flush=True)
    total_elapsed = float(prepared.get("prepare_elapsed") or 0.0) + render_elapsed
    print(f"  [Page {page_index+1}] TOTAL: {total_elapsed:.2f}s", flush=True)
    return result


def process_page(converter, page, page_index: int, pdf_path: Path, assets_dir: Path) -> str:
    prepared = prepare_page_render_input(
        converter,
        page,
        page_index=page_index,
        pdf_path=pdf_path,
        assets_dir=assets_dir,
    )
    return render_prepared_page(
        converter,
        prepared=prepared,
        page=page,
        page_index=page_index,
        assets_dir=assets_dir,
    )
