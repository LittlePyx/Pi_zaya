from __future__ import annotations

import re
from typing import Optional, List, Tuple, Set
from collections import Counter

try:
    import fitz
except ImportError:
    fitz = None

from .text_utils import _normalize_text
from .geometry_utils import _bbox_width, _rect_area, _rect_intersection_area, _overlap_1d, _union_rect


def detect_body_font_size(doc) -> float:
    sizes: list[float] = []
    for page in doc:
        d = page.get_text("dict")
        for b in d.get("blocks", []):
            for l in b.get("lines", []) or []:
                for s in l.get("spans", []) or []:
                    try:
                        sizes.append(round(float(s.get("size", 0.0)), 1))
                    except Exception:
                        continue
    if not sizes:
        return 10.0
    return float(Counter(sizes).most_common(1)[0][0])


def build_repeated_noise_texts(doc) -> set[str]:
    """
    Identify text that repeats on almost every page (headers/footers).
    """
    line_counts: Counter[str] = Counter()
    total_pages = len(doc)
    sample_pages = range(total_pages)
    if total_pages > 20:
        # Sample first 3, last 3, and some middle
        sample_pages = list(range(3)) + list(range(total_pages - 3, total_pages)) + list(range(10, 15))
        sample_pages = sorted(list(set(p for p in sample_pages if 0 <= p < total_pages)))

    for i in sample_pages:
        page = doc[i]
        text = page.get_text("text")
        # Normalize lines
        seen_on_page = set()
        for line in text.splitlines():
            t = _normalize_text(line).strip()
            if len(t) < 4:
                continue
            # Avoid aggressive filtering of short numbers (page numbers are handled by easy regex anyway)
            # but usually headers are longer.
            key = t
            if key in seen_on_page:
                continue
            seen_on_page.add(key)
            line_counts[key] += 1

    # Threshold: appears on > 60% of sampled pages?
    limit = max(2, int(len(sample_pages) * 0.6))
    noise = {line for line, count in line_counts.items() if count >= limit}
    return noise


def _is_frontmatter_noise_line(text: str) -> bool:
    t = _normalize_text(text)
    if not t:
        return True
    front_pat = [
        r"^Latest updates:\s*",
        r"^PDF Download\b",
        r"^Total Citations\b",
        r"^Total Downloads\b",
        r"^Published:\b",
        r"^Citation in BibTeX format\b",
        r"^Open Access Support provided by:\b",
        r"^RESEARCH-ARTICLE\b",
    ]
    return any(re.match(p, t, flags=re.IGNORECASE) for p in front_pat)


def _collect_image_rects(page) -> list["fitz.Rect"]:
    if fitz is None:
        return []
    out: list[fitz.Rect] = []
    for info in (page.get_image_info() or []):
        if "bbox" not in info:
            continue
        try:
            r = fitz.Rect(info["bbox"])
        except Exception:
            continue
        if r.width <= 1.0 or r.height <= 1.0:
            continue
        out.append(r)
    return out


def _merge_nearby_visual_rects(rects: list["fitz.Rect"], *, page_w: float, page_h: float) -> list["fitz.Rect"]:
    if fitz is None or (not rects):
        return rects
    page_area = max(1.0, float(page_w) * float(page_h))
    merged = [fitz.Rect(r) for r in rects]
    changed = True
    while changed:
        changed = False
        out: list[fitz.Rect] = []
        used = [False] * len(merged)
        for i, a in enumerate(merged):
            if used[i]:
                continue
            cur = fitz.Rect(a)
            used[i] = True
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                b = merged[j]
                x_gap = max(0.0, max(float(cur.x0), float(b.x0)) - min(float(cur.x1), float(b.x1)))
                y_gap = max(0.0, max(float(cur.y0), float(b.y0)) - min(float(cur.y1), float(b.y1)))
                x_ov = _overlap_1d(float(cur.x0), float(cur.x1), float(b.x0), float(b.x1))
                y_ov = _overlap_1d(float(cur.y0), float(cur.y1), float(b.y0), float(b.y1))
                y_min = max(1.0, min(float(cur.height), float(b.height)))
                x_min = max(1.0, min(float(cur.width), float(b.width)))
                close_h = (y_ov / y_min >= 0.45) and (x_gap <= page_w * 0.035)
                close_v = (x_ov / x_min >= 0.45) and (y_gap <= page_h * 0.028)
                touching = _rect_intersection_area(cur, b) > 0.0
                union = fitz.Rect(
                    min(float(cur.x0), float(b.x0)),
                    min(float(cur.y0), float(b.y0)),
                    max(float(cur.x1), float(b.x1)),
                    max(float(cur.y1), float(b.y1)),
                )
                union_area = max(0.0, float(union.width) * float(union.height))
                area_a = max(0.0, float(cur.width) * float(cur.height))
                area_b = max(0.0, float(b.width) * float(b.height))
                # Multi-panel figures are often stacked (a)(b)/(c) with a moderate y-gap.
                # Keep this merge conservative to avoid collapsing unrelated figures.
                stacked_panel = (
                    (x_ov / x_min >= 0.62)
                    and (y_gap <= page_h * 0.085)
                    and (min(area_a, area_b) >= page_area * 0.006)
                    and (union_area <= page_area * 0.62)
                )
                if close_h or close_v or touching or stacked_panel:
                    cur = fitz.Rect(
                        min(float(cur.x0), float(b.x0)),
                        min(float(cur.y0), float(b.y0)),
                        max(float(cur.x1), float(b.x1)),
                        max(float(cur.y1), float(b.y1)),
                    )
                    used[j] = True
                    changed = True
            out.append(cur)
        merged = out
    return merged


def _collect_visual_rects(page, *, image_rects: Optional[list["fitz.Rect"]] = None) -> list["fitz.Rect"]:
    """
    Collect visual regions that may represent figures/charts:
    - embedded image bboxes
    - large vector drawing bboxes (for plots not embedded as images)
    """
    if fitz is None:
        return []

    page_w = float(page.rect.width)
    page_h = float(page.rect.height)
    page_area = max(1.0, page_w * page_h)
    out: list[fitz.Rect] = []

    out.extend([fitz.Rect(r) for r in (image_rects if image_rects is not None else _collect_image_rects(page))])

    # Vector charts/diagrams may not appear in get_image_info().
    try:
        for d in (page.get_drawings() or []):
            r0 = d.get("rect")
            if not r0:
                continue
            try:
                r = fitz.Rect(r0)
            except Exception:
                continue
            area = max(0.0, float(r.width) * float(r.height))
            if area < page_area * 0.003:
                continue
            if area > page_area * 0.92:
                continue
            if float(r.width) < page_w * 0.12 or float(r.height) < page_h * 0.05:
                continue
            # Skip decorative rules.
            if float(r.height) < page_h * 0.02 and float(r.width) > page_w * 0.7:
                continue
            out.append(r)
    except Exception:
        pass

    # De-duplicate near-identical rectangles.
    seen: set[tuple[int, int, int, int]] = set()
    uniq: list[fitz.Rect] = []
    for r in out:
        key = (
            int(round(float(r.x0) * 2)),
            int(round(float(r.y0) * 2)),
            int(round(float(r.x1) * 2)),
            int(round(float(r.y1) * 2)),
        )
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    return _merge_nearby_visual_rects(uniq, page_w=page_w, page_h=page_h)


def _pick_render_scale(page_rect: "fitz.Rect", crop_rect: "fitz.Rect", *, base_scale: float, min_scale: float = 1.35) -> float:
    page_area = max(1.0, float(page_rect.width) * float(page_rect.height))
    crop_area = max(1.0, float(crop_rect.width) * float(crop_rect.height))
    ratio = crop_area / page_area
    scale = float(base_scale)
    if ratio >= 0.58:
        scale = float(base_scale) * 0.62
    elif ratio >= 0.38:
        scale = float(base_scale) * 0.74
    elif ratio >= 0.22:
        scale = float(base_scale) * 0.86
    return max(float(min_scale), min(float(base_scale), float(scale)))


def _pick_column_range(rect: fitz.Rect, page_width: float) -> tuple[float, float]:
    mid = page_width / 2.0
    if rect.width >= page_width * 0.55:
        return (0.0, page_width)
    return (0.0, mid) if rect.x0 < mid else (mid, page_width)


def _detect_column_split_x(blocks: list, page_width: float) -> Optional[float]:
    """
    Detect the x-position separating left/right columns.
    Returns None for likely single-column layouts.
    Uses generic 'blocks' which can be TextBlock objects or objects with .bbox.
    """
    if not blocks:
        return None

    # Assuming blocks have .bbox
    candidates = [
        (float(b.bbox[0]) + float(b.bbox[2])) / 2.0
        for b in blocks
        if _bbox_width(b.bbox) < page_width * 0.62
    ]
    if len(candidates) < 4:
        return None
    centers = sorted(candidates)

    best_gap = 0.0
    best_mid = None
    lo = page_width * 0.22
    hi = page_width * 0.78
    for i in range(len(centers) - 1):
        a, b = centers[i], centers[i + 1]
        mid = (a + b) / 2.0
        if mid < lo or mid > hi:
            continue
        gap = float(b - a)
        if gap > best_gap:
            best_gap = gap
            best_mid = mid

    if best_mid is None or best_gap < page_width * 0.08:
        return None

    left_n = sum(1 for c in centers if c < best_mid)
    right_n = len(centers) - left_n
    if left_n < 2 or right_n < 2:
        return None
    return float(best_mid)


def sort_blocks_reading_order(blocks: list, page_width: float) -> list:
    if not blocks:
        return []

    col_split = _detect_column_split_x(blocks, page_width=page_width)
    if col_split is None:
        return sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))

    spanning_threshold = page_width * 0.62
    cross_margin = max(8.0, page_width * 0.015)
    by_y = sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))

    out: list = []
    segment: list = []

    def flush_segment():
        nonlocal segment
        if not segment:
            return
        left = [b for b in segment if ((float(b.bbox[0]) + float(b.bbox[2])) / 2.0) < col_split]
        right = [b for b in segment if ((float(b.bbox[0]) + float(b.bbox[2])) / 2.0) >= col_split]
        left.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
        right.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
        out.extend(left)
        out.extend(right)
        segment = []

    for b in by_y:
        x0, _, x1, _ = b.bbox
        crosses_split = float(x0) < (col_split - cross_margin) and float(x1) > (col_split + cross_margin)
        if _bbox_width(b.bbox) >= spanning_threshold or crosses_split:
            flush_segment()
            out.append(b)
        else:
            segment.append(b)

    flush_segment()
    return out
