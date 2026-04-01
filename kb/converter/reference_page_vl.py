from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

try:
    import fitz
except ImportError:
    fitz = None

from .text_utils import _normalize_text


def is_reference_placeholder_line(text: str) -> bool:
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


def sanitize_reference_crop_markdown(md: str) -> str:
    if not md:
        return ""
    out: list[str] = []
    for ln in (md or "").splitlines():
        s = (ln or "").strip()
        if not s:
            continue
        if is_reference_placeholder_line(s):
            continue
        if s.startswith("```"):
            continue
        s = s.replace("$$", "").replace("$", "").strip()
        if re.match(r"^#{1,6}\s+", s):
            s = re.sub(r"^#{1,6}\s+", "", s).strip()
        out.append(s)
    return "\n".join(out).strip()


def merge_reference_crop_markdowns(parts: list[str]) -> str:
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
            if is_reference_placeholder_line(s):
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


def build_reference_column_crop_rects(*, page, page_w: float, page_h: float) -> list["fitz.Rect"]:
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


def choose_reference_crop_max_tokens_override(*, speed_mode: str) -> int | None:
    try:
        raw = str(os.environ.get("KB_PDF_VISION_REFS_CROP_MAX_TOKENS", "") or "").strip()
        if raw:
            value = int(raw)
            if value <= 0:
                return None
            return max(1024, min(3072, value))
    except Exception:
        pass

    mode = str(speed_mode or "").strip().lower()
    if mode == "ultra_fast":
        return 1024
    if mode == "normal":
        return 1536
    return None


def convert_references_page_with_column_vl(
    converter,
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
    if fitz is None or page is None:
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
            dpi = int((getattr(converter, "_active_speed_config", None) or {}).get("dpi", 220) or 220)
        except Exception:
            dpi = int(getattr(converter, "dpi", 220) or 220)
    dpi = max(240, min(500, int(dpi)))

    rects = build_reference_column_crop_rects(
        page=page,
        page_w=float(page_w),
        page_h=float(page_h),
    )
    crop_max_tokens = choose_reference_crop_max_tokens_override(speed_mode=speed_mode)

    try:
        print(
            f"[VISION_DIRECT][REFS] page {page_index+1}: column mode enabled "
            f"({len(rects)} crops, dpi={int(dpi)}, max_tokens={crop_max_tokens if crop_max_tokens is not None else 'default'})",
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
        part_md = converter.llm_worker.call_llm_page_to_markdown(
            part_png,
            page_number=page_index,
            total_pages=total_pages,
            hint=col_hint,
            speed_mode=speed_mode,
            is_references_page=True,
            max_tokens_override=crop_max_tokens,
        )
        part_md = sanitize_reference_crop_markdown((part_md or "").strip())
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
    merged = merge_reference_crop_markdowns(parts)
    return merged or None
