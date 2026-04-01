from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

try:
    import fitz
except ImportError:
    fitz = None

from .layout_analysis import _detect_column_split_x, detect_body_font_size
from .models import TextBlock
from .tables import _extract_tables_by_layout, _page_maybe_has_table_from_dict
from .text_utils import _normalize_text


def is_probable_page_number_block(block: TextBlock, *, page_w: float, page_h: float) -> bool:
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


def structured_crop_lane_for_bbox(
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


def expand_structured_crop_rect(
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


def fallback_markdown_from_blocks(blocks: list[TextBlock]) -> str:
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


def _build_structured_crop_regions(
    converter,
    *,
    blocks: list[TextBlock],
    page_w: float,
    page_h: float,
    col_split: float,
) -> list[dict]:
    annotated: list[dict] = []
    for block in blocks:
        try:
            bbox = tuple(float(v) for v in block.bbox)
        except Exception:
            continue
        lane = structured_crop_lane_for_bbox(
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
        return []

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
            region_blocks = [row["block"] for row in lane_rows]
            rect = expand_structured_crop_rect(
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
                    "blocks": region_blocks,
                    "fallback_md": fallback_markdown_from_blocks(region_blocks),
                }
            )

    for fg in sorted(full_group_rects, key=lambda row: (float(row["rect"].y0), float(row["rect"].x0))):
        _append_interval_regions(cursor_y, float(fg["rect"].y0) - 2.0)
        order_idx += 1
        fg_blocks = [row["block"] for row in fg["blocks"]]
        regions.append(
            {
                "order": order_idx,
                "lane": "full",
                "rect": expand_structured_crop_rect(
                    fg["rect"],
                    lane="full",
                    page_w=page_w,
                    page_h=page_h,
                    col_split=float(col_split),
                ),
                "blocks": fg_blocks,
                "fallback_md": fallback_markdown_from_blocks(fg_blocks),
            }
        )
        cursor_y = float(fg["rect"].y1) + 2.0

    _append_interval_regions(cursor_y, float(page_h))
    return [row for row in regions if row.get("fallback_md") or row.get("blocks")]


def _resolve_layout_crop_dpi(converter) -> int:
    try:
        raw_dpi = str(os.environ.get("KB_PDF_VISION_DPI", "") or "").strip()
        dpi = int(raw_dpi) if raw_dpi else 0
    except Exception:
        dpi = 0
    if dpi <= 0:
        try:
            dpi = int((getattr(converter, "_active_speed_config", None) or {}).get("dpi", 220) or 220)
        except Exception:
            dpi = int(getattr(converter, "dpi", 220) or 220)
    return max(220, min(420, int(dpi)))


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


def _merge_missing_tables_back_into_region_markdown(md_part: str, *, blocks: list[TextBlock]) -> str:
    table_rows: list[tuple[int, str]] = []
    for pos, block in enumerate(blocks or []):
        if bool(getattr(block, "is_table", False)) and getattr(block, "table_markdown", None):
            table_rows.append((pos, str(block.table_markdown or "").strip()))
    if not table_rows:
        return md_part

    md_norm = re.sub(r"\s+", " ", _normalize_text(md_part or "")).strip().lower()
    missing_tables: list[tuple[int, str]] = []
    for pos, table_md in table_rows:
        key = _table_presence_key(table_md)
        if key and key in md_norm:
            continue
        missing_tables.append((pos, table_md))
    if not missing_tables:
        return md_part

    first_non_table_idx = next(
        (
            idx for idx, block in enumerate(blocks or [])
            if not bool(getattr(block, "is_table", False))
        ),
        len(blocks or []),
    )
    prefix_tables = [table_md for pos, table_md in missing_tables if pos <= first_non_table_idx]
    suffix_tables = [table_md for pos, table_md in missing_tables if pos > first_non_table_idx]
    return "\n\n".join(
        part for part in (
            *(prefix_tables or []),
            md_part,
            *(suffix_tables or []),
        )
        if part
    ).strip()


def _is_substantial_text_block(block: TextBlock, *, page_w: float) -> bool:
    try:
        x0, y0, x1, y1 = (float(v) for v in block.bbox)
    except Exception:
        return False
    width = max(0.0, x1 - x0)
    height = max(0.0, y1 - y0)
    if width < page_w * 0.18 or width >= page_w * 0.62:
        return False
    if height < 34.0:
        return False
    text = _normalize_text(str(getattr(block, "text", "") or "")).strip()
    if len(text) < 36:
        return False
    if bool(getattr(block, "is_table", False)) or bool(getattr(block, "is_math", False)):
        return False
    return True


def _is_probable_running_header_footer_block(block: TextBlock, *, page_w: float, page_h: float) -> bool:
    try:
        x0, y0, x1, y1 = (float(v) for v in block.bbox)
    except Exception:
        return False
    text = _normalize_text(str(getattr(block, "text", "") or "")).strip()
    if not text:
        return False
    is_top_or_bottom_edge = float(y1) <= page_h * 0.09 or float(y0) >= page_h * 0.94
    if not is_top_or_bottom_edge:
        return False
    width = max(0.0, x1 - x0)
    if width < page_w * 0.55:
        return False
    if re.search(r"\bpage\s+\d+\s+of\s+\d+\b", text, flags=re.IGNORECASE):
        return True
    if re.search(r"\blight:\s*science\s*&\s*applications\b", text, flags=re.IGNORECASE):
        return True
    return False


def _looks_like_section_opening_text(text: str) -> bool:
    t = _normalize_text(text or "").strip()
    if not t:
        return False
    return bool(
        re.match(
            r"^(?:results?|discussion|materials?\s+and\s+methods?|methods?|conclusion|experimental|principle)\b",
            t,
            flags=re.IGNORECASE,
        )
    )


def _has_cross_column_reading_order_risk(
    blocks: list[TextBlock],
    *,
    page_w: float,
    page_h: float,
    col_split: float,
) -> bool:
    left_blocks: list[TextBlock] = []
    right_blocks: list[TextBlock] = []
    for block in blocks or []:
        if not _is_substantial_text_block(block, page_w=page_w):
            continue
        cx = (float(block.bbox[0]) + float(block.bbox[2])) * 0.5
        if cx < float(col_split):
            left_blocks.append(block)
        else:
            right_blocks.append(block)
    if not left_blocks or not right_blocks:
        return False

    same_row_tol = max(18.0, page_h * 0.025)
    for left in left_blocks:
        left_text = _normalize_text(str(getattr(left, "text", "") or "")).strip()
        if _looks_like_section_opening_text(left_text):
            continue
        ly0, ly1 = float(left.bbox[1]), float(left.bbox[3])
        for right in right_blocks:
            ry0, ry1 = float(right.bbox[1]), float(right.bbox[3])
            if abs(ly0 - ry0) > same_row_tol:
                continue
            y_overlap = min(ly1, ry1) - max(ly0, ry0)
            if y_overlap < 28.0:
                continue
            right_text = _normalize_text(str(getattr(right, "text", "") or "")).strip()
            if _looks_like_section_opening_text(right_text):
                return True
            if bool(getattr(right, "heading_level", None)) and (not bool(getattr(left, "heading_level", None))):
                return True
    return False


def _ocr_layout_region(
    converter,
    *,
    page,
    page_index: int,
    total_pages: int,
    page_hint: str,
    speed_mode: str,
    dpi: int,
    region_count: int,
    row: dict,
) -> tuple[int, str]:
    rid = int(row["order"])
    lane = str(row["lane"])
    rect = fitz.Rect(row["rect"])
    blocks = list(row.get("blocks") or [])
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
        + f"This is structured crop {rid}/{region_count} from a paper page. "
          f"It covers the {lane} layout region only. "
          "Output only the content visible in this crop, in normal reading order, as Markdown. "
          "Preserve formulas, tables, captions, and headings exactly when present. "
          "Do not invent or repeat content from other regions."
    )
    t0 = time.time()
    md_part = converter.llm_worker.call_llm_page_to_markdown(
        png,
        page_number=page_index,
        total_pages=total_pages,
        hint=hint,
        speed_mode=speed_mode,
        is_references_page=False,
    )
    md_part = (md_part or "").strip() or fallback_md
    md_part = _merge_missing_tables_back_into_region_markdown(md_part, blocks=blocks)
    try:
        print(
            f"[VISION_DIRECT][LAYOUT] page {page_index+1} crop {rid}/{region_count} ({lane}) done ({time.time()-t0:.1f}s, {len(md_part)} chars)",
            flush=True,
        )
    except Exception:
        pass
    return rid, md_part


def convert_page_with_layout_crops(
    converter,
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
    manual_mode = bool(converter._vision_layout_crop_mode_enabled())

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
    tables = []
    if has_table_hint:
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

    try:
        body_size = detect_body_font_size([page])
    except Exception:
        body_size = 10.0

    try:
        blocks = converter._extract_text_blocks(
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
        if not is_probable_page_number_block(b, page_w=page_w, page_h=page_h)
        and not _is_probable_running_header_footer_block(b, page_w=page_w, page_h=page_h)
    ]
    if len(blocks) < 3:
        return None

    col_split = _detect_column_split_x(blocks, page_width=page_w)
    if col_split is None:
        return None

    cross_column_risk = _has_cross_column_reading_order_risk(
        blocks,
        page_w=page_w,
        page_h=page_h,
        col_split=float(col_split),
    )
    if (not tables) and (not manual_mode) and (not cross_column_risk):
        return None

    regions = _build_structured_crop_regions(
        converter,
        blocks=blocks,
        page_w=page_w,
        page_h=page_h,
        col_split=float(col_split),
    )
    if len(regions) <= 1:
        return None

    dpi = _resolve_layout_crop_dpi(converter)
    try:
        print(
            f"[VISION_DIRECT][LAYOUT] page {page_index+1}: structured crop mode enabled "
            f"({len(regions)} crops, dpi={int(dpi)}, tables={int(bool(tables))}, risk={int(bool(cross_column_risk))})",
            flush=True,
        )
    except Exception:
        pass

    ordered_md: dict[int, str] = {}
    region_count = len(regions)
    max_workers = min(4, max(1, region_count))
    if max_workers <= 1:
        for row in regions:
            rid, md_part = _ocr_layout_region(
                converter,
                page=page,
                page_index=page_index,
                total_pages=total_pages,
                page_hint=page_hint,
                speed_mode=speed_mode,
                dpi=dpi,
                region_count=region_count,
                row=row,
            )
            if md_part:
                ordered_md[rid] = md_part
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = [
                pool.submit(
                    _ocr_layout_region,
                    converter,
                    page=page,
                    page_index=page_index,
                    total_pages=total_pages,
                    page_hint=page_hint,
                    speed_mode=speed_mode,
                    dpi=dpi,
                    region_count=region_count,
                    row=row,
                )
                for row in regions
            ]
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
