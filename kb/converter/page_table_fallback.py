from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

try:
    import fitz
except ImportError:
    fitz = None

from .geometry_utils import _rect_area, _rect_intersection_area, _overlap_1d
from .tables import _page_maybe_has_table_from_dict
from .text_utils import _normalize_text


def detect_table_rects_for_fallback(
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


def inject_table_image_fallbacks(
    md: str,
    *,
    page,
    page_index: int,
    pdf_path: Path,
    assets_dir: Path,
    visual_rects: Optional[list["fitz.Rect"]] = None,
    is_references_page: bool = False,
    dpi: int = 200,
) -> str:
    """
    For difficult tables that VL fails to render into markdown, insert a table
    screenshot fallback (saved as `page_X_table_Y.png`) before the table caption.
    """
    if fitz is None or not md or is_references_page:
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

    table_rects = detect_table_rects_for_fallback(
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
    dpi = max(144, int(dpi or 200))

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
