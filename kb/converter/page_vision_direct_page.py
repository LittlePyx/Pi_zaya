from __future__ import annotations

import os
import re
import time
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

try:
    import fitz
except ImportError:
    fitz = None

from .geometry_utils import _bbox_width, _overlap_1d, _rect_area
from .figure_assets import figure_asset_needs_refresh, resolve_figure_asset_dpi
from .heuristics import _page_is_predominantly_references
from .layout_analysis import _collect_visual_rects, _detect_column_split_x, page_has_full_page_image_layer
from .page_figure_metadata import infer_visual_rects_from_caption_candidates
from .reference_page_vl import reference_markdown_entry_count, reference_markdown_is_usable
from .tables import _page_maybe_has_table_from_dict


_PAGE_CLASS_REFERENCES = "references"
_PAGE_CLASS_TEXT_DENSE = "text_dense_body"
_PAGE_CLASS_VISUAL = "figure_or_visual_heavy"
_PAGE_CLASS_FORMULA = "formula_sensitive"
_PAGE_CLASS_UNKNOWN = "unknown"


def _adaptive_page_budgets_enabled() -> bool:
    """Return whether the new class-aware budget policy is enabled.

    The existing normal-mode plain-page budgets remain the compatibility path.
    This flag only enables the stricter class-aware policy, so rollout and A/B
    measurements can happen without silently changing production conversion.
    """
    try:
        raw = str(os.environ.get("KB_PDF_VISION_ADAPTIVE_PAGE_BUDGETS", "0") or "0").strip().lower()
    except Exception:
        return False
    return raw in {"1", "true", "yes", "y", "on"}


def _vision_text_local_fastpath_enabled() -> bool:
    """Return whether verified text-only pages may bypass the vision model."""
    try:
        raw = str(os.environ.get("KB_PDF_VISION_TEXT_LOCAL_FASTPATH", "0") or "0").strip().lower()
    except Exception:
        return False
    return raw in {"1", "true", "yes", "y", "on"}


def _page_has_table_risk(page) -> bool:
    try:
        page_dict = page.get_text("dict") or {}
    except Exception:
        return True
    if not isinstance(page_dict, dict) or not page_dict:
        return True
    try:
        return bool(_page_maybe_has_table_from_dict(page_dict))
    except Exception:
        return True


def _page_looks_like_reference_continuation(page) -> bool:
    """Reject heading-less bibliography continuation pages from the body fast path."""
    try:
        text = str(page.get_text("text") or "")
    except Exception:
        return True
    reference_numbers = [
        int(value)
        for value in re.findall(r"(?m)^\s*\[\s*(\d{1,4})\s*\]\s+\S", text)
    ]
    if len(reference_numbers) < 3:
        return False
    return all(current > previous for previous, current in zip(reference_numbers, reference_numbers[1:]))


def _local_text_fastpath_eligibility(
    converter,
    *,
    page,
    speed_mode: str,
    page_class: str,
    plain_text_chars: int,
) -> tuple[bool, str]:
    if str(speed_mode or "").strip().lower() != "normal":
        return False, "speed_mode"
    if page_class != _PAGE_CLASS_TEXT_DENSE:
        return False, "page_class"
    if int(plain_text_chars or 0) < 900:
        return False, "source_too_short"
    if not callable(getattr(converter, "_process_page_local_only", None)):
        return False, "local_pipeline_unavailable"
    if _page_looks_like_reference_continuation(page):
        return False, "references_continuation"
    if _page_has_table_risk(page):
        return False, "table_risk"
    return True, "eligible"


def _normalized_word_tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return re.findall(r"[^\W_]+(?:['’\-][^\W_]+)*", normalized, flags=re.UNICODE)


def _counter_recall(source_items: list, output_items: list) -> float:
    if not source_items:
        return 0.0
    source_counts = Counter(source_items)
    output_counts = Counter(output_items)
    matched = sum(min(count, int(output_counts.get(item, 0))) for item, count in source_counts.items())
    return float(matched) / float(len(source_items))


def _source_bold_prefix_headings(page, *, noise_texts: Optional[set[str]] = None) -> list[str]:
    try:
        page_dict = page.get_text("dict") or {}
    except Exception:
        return []
    headings: list[str] = []
    seen: set[str] = set()
    normalized_noise = {" ".join(_normalized_word_tokens(item)) for item in (noise_texts or set())}
    for block in page_dict.get("blocks") or []:
        if int(block.get("type") or 0) != 0:
            continue
        spans = [
            span
            for line in block.get("lines") or []
            for span in line.get("spans") or []
            if str(span.get("text") or "").strip()
        ]
        if not spans:
            continue
        bold_parts: list[str] = []
        for span in spans:
            font = str(span.get("font") or "").strip().lower()
            try:
                bold_flag = bool(int(span.get("flags") or 0) & 16)
            except Exception:
                bold_flag = False
            if not (bold_flag or any(label in font for label in ("bold", "semibold", "demibold"))):
                break
            bold_parts.append(str(span.get("text") or "").strip())
        if not bold_parts:
            continue
        heading = re.sub(r"\s+", " ", " ".join(bold_parts)).strip(" -–—:;")
        words = _normalized_word_tokens(heading)
        if len(heading) < 4 or len(heading) > 120 or not (1 <= len(words) <= 16):
            continue
        if heading.endswith((".", ",")) or not re.search(r"[^\W\d_]", heading, flags=re.UNICODE):
            continue
        key = " ".join(words)
        if key in {"article", "open access"} or key in normalized_noise:
            continue
        if not key or key in seen:
            continue
        seen.add(key)
        headings.append(heading)
    return headings


def _promote_source_headings(markdown: str | None, headings: list[str]) -> tuple[str, int]:
    output = str(markdown or "")
    promoted = 0
    for heading in sorted(headings, key=len, reverse=True):
        parts = [re.escape(part) for part in re.split(r"\s+", heading.strip()) if part]
        if not parts:
            continue
        flexible = r"[ \t]+".join(parts)
        pattern = re.compile(rf"(?m)^(?![ \t]*#)(?P<lead>[ \t]*){flexible}(?=[ \t]+)", flags=re.UNICODE)

        def _replacement(match: re.Match) -> str:
            return f"{match.group('lead')}### {heading}\n\n"

        output, count = pattern.subn(_replacement, output, count=1)
        promoted += int(count)
    return output, promoted


def _validate_local_text_markdown(
    source_text: str,
    markdown: str | None,
    *,
    required_headings: Optional[list[str]] = None,
) -> tuple[bool, str, dict[str, float | int]]:
    source_raw = str(source_text or "")
    output_raw = str(markdown or "").strip()
    source_tokens = _normalized_word_tokens(source_raw)
    output_tokens = _normalized_word_tokens(output_raw)
    metrics: dict[str, float | int] = {
        "source_chars": len(source_raw.strip()),
        "output_chars": len(output_raw),
        "source_tokens": len(source_tokens),
        "output_tokens": len(output_tokens),
        "coverage": 0.0,
        "bigram_coverage": 0.0,
        "order_ratio": 0.0,
        "length_ratio": 0.0,
        "heading_candidates": len(required_headings or []),
        "headings_promoted": 0,
    }
    if len(source_tokens) < 120:
        return False, "source_tokens_too_few", metrics
    if len(output_tokens) < 100:
        return False, "output_too_short", metrics
    if re.search(r"\[Page\s+\d+\s+conversion\s+incomplete\]", output_raw, flags=re.IGNORECASE):
        return False, "incomplete_output", metrics
    if "\ufffd" in output_raw or any(ord(ch) < 32 and ch not in "\n\r\t" for ch in output_raw):
        return False, "garbled_output", metrics

    coverage = _counter_recall(source_tokens, output_tokens)
    source_bigrams = list(zip(source_tokens, source_tokens[1:]))
    output_bigrams = list(zip(output_tokens, output_tokens[1:]))
    bigram_coverage = _counter_recall(source_bigrams, output_bigrams)
    order_ratio = SequenceMatcher(None, source_tokens, output_tokens).ratio()
    length_ratio = float(len(output_tokens)) / float(max(1, len(source_tokens)))
    metrics.update(
        {
            "coverage": coverage,
            "bigram_coverage": bigram_coverage,
            "order_ratio": order_ratio,
            "length_ratio": length_ratio,
        }
    )
    if length_ratio < 0.72:
        return False, "length_too_short", metrics
    if length_ratio > 1.35:
        return False, "length_too_long", metrics
    if coverage < 0.84:
        return False, "low_word_coverage", metrics
    if bigram_coverage < 0.68:
        return False, "low_bigram_coverage", metrics
    if order_ratio < 0.62:
        return False, "low_order_similarity", metrics
    markdown_heading_keys = {
        " ".join(_normalized_word_tokens(line))
        for line in re.findall(r"(?m)^#{1,6}\s+(.+?)\s*$", output_raw)
    }
    for heading in required_headings or []:
        heading_key = " ".join(_normalized_word_tokens(heading))
        if heading_key and heading_key not in markdown_heading_keys:
            return False, "missing_source_heading", metrics
    return True, "accepted", metrics


def _collect_page_formula_candidates(
    converter,
    *,
    page,
    page_index: int,
    is_references_page: bool,
) -> Optional[list]:
    if is_references_page:
        return []
    collector = getattr(converter, "_collect_display_math_candidates", None)
    if not callable(collector):
        return None
    try:
        candidates = collector(
            page,
            page_index=int(page_index),
            is_references_page=False,
        )
    except Exception:
        return None
    return list(candidates or [])


def _classify_page_budget(
    *,
    page_index: int,
    is_references_page: bool,
    image_names: list[str],
    visual_rects: list["fitz.Rect"],
    formula_candidate_count: int,
    plain_text_chars: int,
) -> str:
    """Classify a page using only source-local signals available before VL."""
    if is_references_page:
        return _PAGE_CLASS_REFERENCES
    if int(formula_candidate_count or 0) > 0:
        return _PAGE_CLASS_FORMULA
    if image_names or visual_rects:
        return _PAGE_CLASS_VISUAL
    # Keep cover/front-matter and textless/scan-like pages on the current
    # quality envelope until a reviewed classifier can distinguish them.
    if int(page_index) <= 0 or int(plain_text_chars or 0) <= 0:
        return _PAGE_CLASS_UNKNOWN
    return _PAGE_CLASS_TEXT_DENSE


def _stage_timing_enabled() -> bool:
    try:
        raw = str(os.environ.get("KB_PDF_STAGE_TIMINGS", "") or "").strip().lower()
        return raw in {"1", "true", "yes", "y", "on"}
    except Exception:
        return False


def _detect_references_page(page) -> bool:
    try:
        return bool(_page_is_predominantly_references(page))
    except Exception:
        return False


def _collect_metadata_rects(converter, *, page, page_index: int, is_references_page: bool) -> list["fitz.Rect"]:
    metadata_rects: list[fitz.Rect] = []
    if is_references_page:
        return metadata_rects
    try:
        metadata_rects = converter._collect_non_body_metadata_rects(
            page,
            page_index=page_index,
            is_references_page=False,
        )
    except Exception:
        metadata_rects = []
    return metadata_rects


def _extract_page_visual_assets(
    converter,
    *,
    page,
    page_index: int,
    assets_dir: Path,
    dpi: int,
) -> tuple[list[str], dict[str, dict], list["fitz.Rect"]]:
    image_names: list[str] = []
    figure_entries: list[dict] = []
    figure_meta_by_asset: dict[str, dict] = {}
    visual_rects: list[fitz.Rect] = []
    cap_candidates: list[dict] = []

    try:
        page_w = float(page.rect.width)
        page_h = float(page.rect.height)
        scan_backed_page = page_has_full_page_image_layer(page)
        visual_rects = _collect_visual_rects(page)
        if not visual_rects:
            visual_rects = []
        if visual_rects or scan_backed_page:
            cap_candidates = converter._extract_page_figure_caption_candidates(page)
        W = page_w
        H = page_h
        header_threshold = H * 0.12
        footer_threshold = H * 0.88
        side_margin = W * 0.05
        spanning_threshold = W * 0.55

        def _near_caption(rect: "fitz.Rect") -> bool:
            for cap in cap_candidates:
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
        if (not visual_rects) and scan_backed_page and cap_candidates:
            visual_rects = infer_visual_rects_from_caption_candidates(page, cap_candidates)
        if not visual_rects:
            visual_rects = []
        if visual_rects:
            line_boxes = converter._collect_page_text_line_boxes(page)
        else:
            line_boxes = []

        img_count = 0
        for rect_idx, rect in enumerate(visual_rects):
            is_full_width = _rect_area(rect) >= (W * H * 0.40) or _bbox_width(tuple(rect)) >= spanning_threshold

            if rect.y1 < header_threshold or rect.y0 > footer_threshold:
                if not is_full_width and _rect_area(rect) < (W * H * 0.15):
                    continue

            if not is_full_width:
                if rect.x0 < side_margin and rect.width < W * 0.1:
                    continue
                if rect.x1 > W - side_margin and rect.width < W * 0.1:
                    continue

            cropped_rect = converter._expanded_visual_crop_rect(
                rect=rect,
                page_w=W,
                page_h=H,
                is_full_width=bool(is_full_width),
                line_boxes=line_boxes,
            )

            if cropped_rect.width <= 0 or cropped_rect.height <= 0:
                continue

            img_name = f"page_{page_index+1}_fig_{rect_idx+1}.png"
            img_path = assets_dir / img_name
            used_clip = cropped_rect

            def _record_saved_asset(saved_clip: "fitz.Rect") -> None:
                nonlocal img_count
                img_count += 1
                if img_name not in image_names:
                    image_names.append(img_name)
                figure_entries.append(
                    {
                        "asset_name": img_name,
                        "bbox": [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)],
                        "crop_bbox": [float(saved_clip.x0), float(saved_clip.y0), float(saved_clip.x1), float(saved_clip.y1)],
                    }
                )

            asset_dpi = resolve_figure_asset_dpi(converter, base_dpi=dpi)

            try:
                if not figure_asset_needs_refresh(img_path, clip_rect=used_clip, dpi=asset_dpi):
                    _record_saved_asset(used_clip)
                    continue
            except Exception:
                pass

            try:
                pix_img = page.get_pixmap(clip=cropped_rect, dpi=asset_dpi)
                pix_img.save(img_path)
                if img_path.exists() and img_path.stat().st_size >= 256:
                    _record_saved_asset(used_clip)
            except Exception:
                try:
                    used_clip = rect
                    if not figure_asset_needs_refresh(img_path, clip_rect=used_clip, dpi=asset_dpi):
                        _record_saved_asset(used_clip)
                        continue
                    pix_img = page.get_pixmap(clip=rect, dpi=asset_dpi)
                    pix_img.save(img_path)
                    if img_path.exists() and img_path.stat().st_size >= 256:
                        _record_saved_asset(used_clip)
                except Exception:
                    continue

        if img_count > 0:
            print(f"  [Page {page_index+1}] Extracted {img_count} images", flush=True)
    except Exception as e:
        print(f"  [Page {page_index+1}] Image extraction failed: {e}", flush=True)

    try:
        if figure_entries:
            figure_entries = converter._match_figure_entries_with_captions(
                page=page,
                figure_entries=figure_entries,
                caption_candidates=cap_candidates,
            )
        figure_meta_by_asset = converter._persist_page_figure_metadata(
            assets_dir=assets_dir,
            page_index=page_index,
            figure_entries=figure_entries,
        )
    except Exception as e:
        print(f"  [Page {page_index+1}] Figure metadata persist skipped: {e}", flush=True)
        figure_meta_by_asset = {}

    return image_names, figure_meta_by_asset, visual_rects


def _compress_png_bytes(png_bytes: bytes, *, speed_config: dict) -> bytes:
    try:
        compress_level_raw = os.environ.get("KB_PDF_VISION_COMPRESS", "").strip()
        if compress_level_raw:
            compress_level = int(compress_level_raw)
        else:
            compress_level = speed_config.get("compress", 3)
        if compress_level > 0:
            try:
                from PIL import Image
                import io

                img = Image.open(io.BytesIO(png_bytes))
                output = io.BytesIO()
                img.save(output, format="PNG", optimize=True, compress_level=min(9, max(1, compress_level)))
                png_bytes = output.getvalue()
            except ImportError:
                pass
    except Exception:
        pass
    return png_bytes


def _build_page_hint(
    converter,
    *,
    page_index: int,
    is_references_page: bool,
    image_names: list[str],
    figure_meta_by_asset: dict[str, dict],
) -> str:
    page_hint = ""
    if is_references_page:
        page_hint = (
            "This page is likely in the REFERENCES/BIBLIOGRAPHY section. "
            "Output plain-text references only, one complete reference per line. "
            "Do not use $...$ or $$...$$ on this page."
        )
    else:
        if page_index == 0:
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
        hint_map = converter._build_page_figure_mapping_hint(figure_meta_by_asset)
        if hint_map:
            page_hint = (page_hint + " " + hint_map).strip()
    return page_hint


def _detect_text_column_split_x(page) -> Optional[float]:
    try:
        page_w = float(page.rect.width)
        page_dict = page.get_text("dict")
    except Exception:
        return None

    blocks: list[SimpleNamespace] = []
    for block in (page_dict.get("blocks") or []):
        try:
            bbox = tuple(float(v) for v in (block.get("bbox") or ()))
        except Exception:
            bbox = ()
        if len(bbox) != 4:
            continue
        width = max(0.0, bbox[2] - bbox[0])
        height = max(0.0, bbox[3] - bbox[1])
        if width >= page_w * 0.72 or width <= page_w * 0.12 or height < 24.0:
            continue
        parts: list[str] = []
        for line in block.get("lines") or []:
            spans = line.get("spans") or []
            text = " ".join((s.get("text") or "") for s in spans).strip()
            if text:
                parts.append(text)
        text_blob = " ".join(parts).strip()
        if len(text_blob) < 40:
            continue
        blocks.append(SimpleNamespace(bbox=bbox, text=text_blob))

    if len(blocks) < 2:
        return None
    return _detect_column_split_x(blocks, page_width=page_w)


def _build_layout_page_hint(*, page, visual_rects: list["fitz.Rect"]) -> str:
    hints: list[str] = []
    try:
        page_w = float(page.rect.width)
        page_h = float(page.rect.height)
    except Exception:
        return ""

    col_split = _detect_text_column_split_x(page)
    if col_split is not None:
        hints.append(
            "This page appears to be two-column. Preserve strict reading order for body text: finish the left column completely before starting the right column, even if a new section heading appears near the top of the right column."
        )

    if visual_rects:
        hints.append(
            "Do not output figure-internal panel letters, axis ticks, legends, or short in-image annotations as standalone body lines. Keep such text only when it is part of the visible figure caption."
        )
        has_full_width_mid_figure = any(
            (_bbox_width(tuple(r)) >= page_w * 0.55)
            and (float(r.y0) > page_h * 0.10)
            and (float(r.y1) < page_h * 0.86)
            for r in (visual_rects or [])
        )
        if has_full_width_mid_figure and col_split is not None:
            hints.append(
                "If a wide figure spans both columns in the middle of the page, keep the surrounding text in this order: body text above the figure, then the figure and its caption as one unit, then the body text below the figure. Do not let the figure caption absorb nearby body paragraphs."
            )

    return " ".join(hints).strip()


def _should_prefer_local_figure_order_pipeline(
    *,
    converter,
    page,
    page_index: int,
    image_names: list[str],
    visual_rects: list["fitz.Rect"],
    formula_candidates: Optional[list] = None,
) -> bool:
    if fitz is None or (not image_names) or len(visual_rects) != 1:
        return False
    try:
        page_w = float(page.rect.width)
        page_h = float(page.rect.height)
        rect = fitz.Rect(visual_rects[0])
    except Exception:
        return False

    if float(rect.width) < page_w * 0.68:
        return False
    if float(rect.height) < page_h * 0.30:
        return False
    if float(rect.y0) > page_h * 0.22:
        return False
    if float(rect.y1) < page_h * 0.46:
        return False

    # The local figure-order route is deliberately text-first and can split a
    # display equation into interleaved glyph/prose blocks.  Prefer structured
    # column OCR whenever the source-backed formula detector finds an equation,
    # including formulas whose individual PDF text fragments are too short for
    # the lightweight raw-block heuristic below.
    detected_formula_candidates = formula_candidates
    formula_collector = getattr(converter, "_collect_display_math_candidates", None)
    if detected_formula_candidates is None and callable(formula_collector):
        try:
            detected_formula_candidates = formula_collector(
                page,
                page_index=int(page_index),
                is_references_page=False,
            )
        except Exception:
            pass
    if detected_formula_candidates:
        return False

    try:
        page_dict = page.get_text("dict") or {}
    except Exception:
        page_dict = {}

    below_blocks = 0
    below_chars = 0
    math_like_below = 0
    for block in (page_dict.get("blocks") or []):
        try:
            bbox = fitz.Rect(block.get("bbox"))
        except Exception:
            continue
        if float(bbox.y0) < float(rect.y1) + max(14.0, page_h * 0.015):
            continue
        if float(bbox.width) < page_w * 0.22 or float(bbox.height) < 10.0:
            continue
        parts: list[str] = []
        for line in (block.get("lines") or []):
            for span in (line.get("spans") or []):
                text = str(span.get("text") or "").strip()
                if text:
                    parts.append(text)
        merged = re.sub(r"\s+", " ", " ".join(parts)).strip()
        if len(merged) < 24:
            continue
        below_blocks += 1
        below_chars += len(merged)
        sym_n = len(re.findall(r"[=+\-*/^_{}\\]|[\u0394\u03bb\u03c0\u2211\u222b]", merged))
        word_n = len(re.findall(r"\b[A-Za-z]{2,}\b", merged))
        if sym_n >= 4 and word_n <= 18:
            math_like_below += 1

    return below_blocks >= 2 and below_chars >= 120 and math_like_below == 0


def _should_prefer_local_references_pipeline(converter) -> bool:
    if not hasattr(converter, "_process_page"):
        return False
    try:
        return bool(converter._vision_references_prefer_local_enabled())
    except Exception:
        return True


def _choose_page_max_tokens_override(
    *,
    speed_mode: str,
    page_index: int,
    is_references_page: bool,
    page_hint: str,
    image_names: list[str],
    visual_rects: list["fitz.Rect"],
    formula_placeholders: dict[str, str],
    plain_text_density: str = "unknown",
    page_class: str = "",
    adaptive_enabled: bool = False,
) -> Optional[int]:
    if str(speed_mode or "").strip().lower() != "normal":
        return None
    if is_references_page or page_index <= 0:
        return None
    if adaptive_enabled and page_class != _PAGE_CLASS_TEXT_DENSE:
        return None
    if image_names or visual_rects or formula_placeholders:
        return None
    # Under the adaptive policy, a source-verified text-only page may still carry
    # a harmless layout hint (for example, two-column reading order).  That hint
    # improves reconstruction but does not justify the full generation budget.
    # Legacy behavior remains unchanged while the feature flag is disabled.
    if page_hint and not adaptive_enabled:
        return None
    try:
        raw = str(os.environ.get("KB_PDF_VISION_PLAIN_PAGE_MAX_TOKENS", "") or "").strip()
        if raw:
            value = int(raw)
            if value <= 0:
                return None
            return max(1024, min(4096, value))
    except Exception:
        pass
    # Plain middle-body pages rarely need the full default budget.
    # Keep page 2 slightly more conservative, then trim a bit harder for deeper body pages.
    if plain_text_density == "light":
        if page_index <= 1:
            return 2304
        return 2048
    if page_index <= 1:
        return 2816
    return 2560


def _classify_plain_body_text_density(page) -> tuple[str, int]:
    try:
        raw = str(page.get_text("text") or "")
    except Exception:
        raw = ""
    text = re.sub(r"\s+", " ", raw).strip()
    text_chars = len(text)
    if text_chars <= 0:
        return "unknown", 0
    if text_chars <= 2500:
        return "light", text_chars
    if text_chars <= 4200:
        return "medium", text_chars
    return "dense", text_chars


def _choose_page_render_dpi(
    converter,
    *,
    speed_mode: str,
    page_index: int,
    is_references_page: bool,
    image_names: list[str],
    visual_rects: list["fitz.Rect"],
    base_dpi: int,
    page_class: str = "",
    adaptive_enabled: bool = False,
) -> tuple[int, str]:
    try:
        raw_forced = str(os.environ.get("KB_PDF_VISION_DPI", "") or "").strip()
        if raw_forced and int(raw_forced) > 0:
            return int(base_dpi), "forced"
    except Exception:
        pass

    if str(speed_mode or "").strip().lower() != "normal":
        return int(base_dpi), "base"
    if adaptive_enabled:
        if page_class != _PAGE_CLASS_TEXT_DENSE:
            return int(base_dpi), f"adaptive_{page_class or _PAGE_CLASS_UNKNOWN}"
        try:
            raw_plain = str(os.environ.get("KB_PDF_VISION_PLAIN_PAGE_DPI", "") or "").strip()
            target = int(raw_plain) if raw_plain else 200
        except Exception:
            target = 200
        target = max(200, min(int(base_dpi), int(target)))
        if target < int(base_dpi):
            return int(target), f"adaptive_{_PAGE_CLASS_TEXT_DENSE}"
        return int(base_dpi), f"adaptive_{_PAGE_CLASS_TEXT_DENSE}"
    if is_references_page:
        return int(base_dpi), "references"
    if page_index <= 0:
        return int(base_dpi), "first_page"
    if image_names or visual_rects:
        return int(base_dpi), "visual"
    try:
        if bool(converter._vision_formula_overlay_enabled()):
            return int(base_dpi), "formula_overlay"
    except Exception:
        pass

    try:
        raw_plain = str(os.environ.get("KB_PDF_VISION_PLAIN_PAGE_DPI", "") or "").strip()
        target = int(raw_plain) if raw_plain else 200
    except Exception:
        target = 200
    target = max(200, min(int(base_dpi), int(target)))
    if target < int(base_dpi):
        return int(target), "plain_body"
    return int(base_dpi), "base"


def _apply_formula_overlay(
    converter,
    *,
    png_bytes: bytes,
    page,
    page_index: int,
    page_w: float,
    page_h: float,
    dpi: int,
    is_references_page: bool,
    page_hint: str,
    formula_candidates: Optional[list] = None,
) -> tuple[bytes, str, dict[str, str]]:
    formula_placeholders: dict[str, str] = {}
    if (not is_references_page) and converter._vision_formula_overlay_enabled():
        try:
            eq_candidates = formula_candidates
            if eq_candidates is None:
                eq_candidates = converter._collect_display_math_candidates(
                    page,
                    page_index=page_index,
                    is_references_page=False,
                )
            if eq_candidates:
                eq_labeled_rects, formula_placeholders = converter._extract_formula_placeholders_for_page(
                    page,
                    page_index=page_index,
                    candidates=eq_candidates,
                    base_dpi=int(dpi),
                )
                if eq_labeled_rects and formula_placeholders:
                    png_bytes = converter._mask_rects_with_labels_on_png(
                        png_bytes,
                        labeled_rects=eq_labeled_rects,
                        page_width=page_w,
                        page_height=page_h,
                    )
                    hint_eq = converter._formula_placeholder_hint(formula_placeholders)
                    if hint_eq:
                        page_hint = (page_hint + " " + hint_eq).strip() if page_hint else hint_eq
                    print(
                        f"  [Page {page_index+1}] Formula overlay enabled: {len(formula_placeholders)} display equation(s)",
                        flush=True,
                    )
        except Exception as e:
            print(f"  [Page {page_index+1}] Formula overlay skipped: {e}", flush=True)
    return png_bytes, page_hint, formula_placeholders


def process_vision_direct_page(
    converter,
    *,
    page,
    page_index: int,
    total_pages: int,
    pdf_path: Path,
    assets_dir: Path,
    speed_mode: str,
    speed_config: dict,
    dpi: int,
    mat,
    started_at: Optional[float] = None,
) -> Optional[str]:
    t0 = time.time() if started_at is None else float(started_at)
    perf_t0 = time.perf_counter()
    log_stage_timings = _stage_timing_enabled()

    def _log_stage(step_no: int, label: str, step_started: float, extra: str = "") -> None:
        if not log_stage_timings:
            return
        suffix = f", {extra}" if extra else ""
        print(
            f"  [Page {page_index+1}] Step {step_no} ({label}): {time.perf_counter()-step_started:.2f}s{suffix}",
            flush=True,
        )

    page_w = float(page.rect.width)
    page_h = float(page.rect.height)

    step_start = time.perf_counter()
    is_references_page = _detect_references_page(page)
    _log_stage(1, "refs check", step_start, f"references={int(bool(is_references_page))}")

    step_start = time.perf_counter()
    metadata_rects = _collect_metadata_rects(
        converter,
        page=page,
        page_index=page_index,
        is_references_page=is_references_page,
    )
    _log_stage(2, "metadata rects", step_start, f"count={len(metadata_rects)}")

    step_start = time.perf_counter()
    if is_references_page:
        image_names, figure_meta_by_asset, visual_rects = [], {}, []
        _log_stage(3, "assets", step_start, "skipped=references")
    else:
        image_names, figure_meta_by_asset, visual_rects = _extract_page_visual_assets(
            converter,
            page=page,
            page_index=page_index,
            assets_dir=assets_dir,
            dpi=dpi,
        )
        _log_stage(3, "assets", step_start, f"images={len(image_names)} visuals={len(visual_rects)}")

    adaptive_page_budgets = _adaptive_page_budgets_enabled()
    text_local_fastpath = _vision_text_local_fastpath_enabled()
    plain_text_density, plain_text_chars = _classify_plain_body_text_density(page)
    try:
        formula_overlay_enabled = bool(converter._vision_formula_overlay_enabled())
    except Exception:
        formula_overlay_enabled = False
    formula_candidates: Optional[list] = None
    # Benchmark controls must pay the same cheap classification cost as the
    # candidate so provider timings remain comparable. Normal product runs keep
    # the probe lazy unless formula overlay or adaptive budgets need it.
    if adaptive_page_budgets or text_local_fastpath or formula_overlay_enabled or log_stage_timings:
        formula_candidates = _collect_page_formula_candidates(
            converter,
            page=page,
            page_index=page_index,
            is_references_page=is_references_page,
        )
    page_class = _classify_page_budget(
        page_index=page_index,
        is_references_page=is_references_page,
        image_names=image_names,
        visual_rects=visual_rects,
        formula_candidate_count=len(formula_candidates or []),
        plain_text_chars=plain_text_chars,
    )
    if (adaptive_page_budgets or text_local_fastpath) and formula_candidates is None and page_class == _PAGE_CLASS_TEXT_DENSE:
        page_class = _PAGE_CLASS_UNKNOWN

    if text_local_fastpath and page_class == _PAGE_CLASS_TEXT_DENSE:
        try:
            source_text = str(page.get_text("text") or "")
        except Exception:
            source_text = ""
        eligible, local_reason = _local_text_fastpath_eligibility(
            converter,
            page=page,
            speed_mode=speed_mode,
            page_class=page_class,
            plain_text_chars=plain_text_chars,
        )
        source_tokens = _normalized_word_tokens(source_text)
        source_headings = _source_bold_prefix_headings(
            page,
            noise_texts=set(getattr(converter, "noise_texts", set()) or set()),
        )
        local_metrics: dict[str, float | int] = {
            "source_chars": len(source_text.strip()),
            "output_chars": 0,
            "source_tokens": len(source_tokens),
            "output_tokens": 0,
            "coverage": 0.0,
            "bigram_coverage": 0.0,
            "order_ratio": 0.0,
            "length_ratio": 0.0,
            "heading_candidates": len(source_headings),
            "headings_promoted": 0,
        }
        md_local_text: Optional[str] = None
        local_started = time.perf_counter()
        if eligible:
            try:
                md_local_text = converter._process_page_local_only(
                    page,
                    page_index=page_index,
                    pdf_path=pdf_path,
                    assets_dir=assets_dir,
                )
                md_local_text, headings_promoted = _promote_source_headings(md_local_text, source_headings)
                accepted, local_reason, local_metrics = _validate_local_text_markdown(
                    source_text,
                    md_local_text,
                    required_headings=source_headings,
                )
                local_metrics["headings_promoted"] = int(headings_promoted)
            except Exception as e:
                accepted = False
                local_reason = "local_pipeline_error"
                print(
                    f"[VISION_DIRECT][TEXT_LOCAL] local pipeline failed on page {page_index+1}: {e}",
                    flush=True,
                )
        else:
            accepted = False
        local_elapsed = time.perf_counter() - local_started
        print(
            f"[VISION_DIRECT][TEXT_LOCAL] page {page_index+1}: "
            f"accepted={int(accepted)} reason={local_reason} "
            f"source_chars={int(local_metrics['source_chars'])} output_chars={int(local_metrics['output_chars'])} "
            f"source_tokens={int(local_metrics['source_tokens'])} output_tokens={int(local_metrics['output_tokens'])} "
            f"coverage={float(local_metrics['coverage']):.4f} "
            f"bigram_coverage={float(local_metrics['bigram_coverage']):.4f} "
            f"order_ratio={float(local_metrics['order_ratio']):.4f} "
            f"length_ratio={float(local_metrics['length_ratio']):.4f} "
            f"headings={int(local_metrics['headings_promoted'])}/{int(local_metrics['heading_candidates'])} "
            f"elapsed={local_elapsed:.4f}",
            flush=True,
        )
        if accepted and md_local_text:
            _log_stage(4, "page render", time.perf_counter(), "skipped=text-local-fastpath")
            _log_stage(5, "hints/overlay", time.perf_counter(), "formula=0 hint=0 text_local=1")
            _log_stage(6, "local text convert", local_started, f"text_local=1 chars={len(md_local_text)}")
            if log_stage_timings:
                print(
                    f"[VISION_DIRECT][BUDGET] page {page_index+1}: "
                    f"class={page_class} enabled={int(adaptive_page_budgets)} "
                    f"dpi=0 base_dpi={int(dpi)} max_tokens=default "
                    f"density={plain_text_density} text_chars={plain_text_chars} "
                    "formulas=0 images=0 visuals=0 profile=text_local_fastpath",
                    flush=True,
                )
                print(f"  [Page {page_index+1}] TOTAL: {time.perf_counter() - perf_t0:.2f}s", flush=True)
            elapsed = time.time() - t0
            print(
                f"Finished page {page_index+1}/{total_pages} ({elapsed:.1f}s, {len(md_local_text)} chars, text-local)",
                flush=True,
            )
            return md_local_text

    if is_references_page and _should_prefer_local_references_pipeline(converter):
        step_start = time.perf_counter()
        try:
            md_local_refs = converter._process_page(
                page,
                page_index=page_index,
                pdf_path=pdf_path,
                assets_dir=assets_dir,
            )
        except Exception as e:
            print(
                f"[VISION_DIRECT] local references fastpath failed on page {page_index+1}: {e}",
                flush=True,
            )
            md_local_refs = None
        if md_local_refs and reference_markdown_is_usable(md_local_refs):
            _log_stage(4, "page render", time.perf_counter(), "skipped=refs-local-fastpath")
            _log_stage(5, "hints/overlay", time.perf_counter(), "formula=0 hint=0 refs_local=1")
            _log_stage(6, "convert", step_start, f"refs_local=1 chars={len(md_local_refs)}")
            total_elapsed = time.perf_counter() - perf_t0
            if log_stage_timings:
                print(f"  [Page {page_index+1}] TOTAL: {total_elapsed:.2f}s", flush=True)
            elapsed = time.time() - t0
            print(
                f"Finished page {page_index+1}/{total_pages} ({elapsed:.1f}s, {len(md_local_refs)} chars)",
                flush=True,
            )
            return md_local_refs
        if md_local_refs:
            try:
                entry_count = reference_markdown_entry_count(md_local_refs)
                print(
                    f"[VISION_DIRECT][REFS] local references output too sparse on page {page_index+1} "
                    f"({entry_count} entries); falling back to VL column OCR",
                    flush=True,
                )
            except Exception:
                pass

    references_column_enabled = False
    if is_references_page:
        try:
            references_column_enabled = bool(converter._vision_references_column_mode_enabled())
        except Exception:
            references_column_enabled = False

    references_column_tried = False
    if is_references_page and references_column_enabled:
        references_column_tried = True
        page_hint = _build_page_hint(
            converter,
            page_index=page_index,
            is_references_page=is_references_page,
            image_names=image_names,
            figure_meta_by_asset=figure_meta_by_asset,
        )
        layout_hint = _build_layout_page_hint(page=page, visual_rects=visual_rects)
        if layout_hint:
            page_hint = (page_hint + " " + layout_hint).strip() if page_hint else layout_hint
        step_start = time.perf_counter()
        try:
            md_refs = converter._convert_references_page_with_column_vl(
                page=page,
                page_index=page_index,
                total_pages=total_pages,
                page_hint=page_hint,
                speed_mode=speed_mode,
            )
        except Exception as e:
            print(
                f"[VISION_DIRECT] references column OCR fastpath failed on page {page_index+1}: {e}",
                flush=True,
            )
            md_refs = None
        if md_refs:
            _log_stage(4, "page render", time.perf_counter(), "skipped=refs-column-fastpath")
            _log_stage(
                5,
                "hints/overlay",
                time.perf_counter(),
                f"formula=0 hint={int(bool(page_hint))} refs_fastpath=1",
            )
            _log_stage(6, "vision convert", step_start, f"refs_fastpath=1 chars={len(md_refs)}")
            step_start = time.perf_counter()
            md_refs = converter._postprocess_vision_page_markdown(
                md_refs,
                page=page,
                page_index=page_index,
                pdf_path=pdf_path,
                assets_dir=assets_dir,
                image_names=image_names,
                figure_meta_by_asset=figure_meta_by_asset,
                visual_rects=visual_rects,
                is_references_page=is_references_page,
            )
            _log_stage(7, "postprocess", step_start, f"chars={len(md_refs)}")
            total_elapsed = time.perf_counter() - perf_t0
            if log_stage_timings:
                print(f"  [Page {page_index+1}] TOTAL: {total_elapsed:.2f}s", flush=True)
            elapsed = time.time() - t0
            print(f"Finished page {page_index+1}/{total_pages} ({elapsed:.1f}s, {len(md_refs)} chars)", flush=True)
            return md_refs

    if (not is_references_page) and _should_prefer_local_figure_order_pipeline(
        converter=converter,
        page=page,
        page_index=page_index,
        image_names=image_names,
        visual_rects=visual_rects,
        formula_candidates=formula_candidates,
    ):
        try:
            step_start = time.perf_counter()
            md_local = converter._process_page(
                page,
                page_index=page_index,
                pdf_path=pdf_path,
                assets_dir=assets_dir,
            )
        except Exception as e:
            print(
                f"[VISION_DIRECT] local figure-order fallback failed on page {page_index+1}: {e}",
                flush=True,
            )
            md_local = None
        if md_local:
            _log_stage(4, "page render", time.perf_counter(), "skipped=local-figure-order")
            _log_stage(
                5,
                "hints/overlay",
                time.perf_counter(),
                "formula=0 hint=0 local_figure_order=1",
            )
            _log_stage(6, "vision convert", step_start, f"local_figure_order=1 chars={len(md_local)}")
            total_elapsed = time.perf_counter() - perf_t0
            if log_stage_timings:
                print(f"  [Page {page_index+1}] TOTAL: {total_elapsed:.2f}s", flush=True)
            elapsed = time.time() - t0
            print(f"Finished page {page_index+1}/{total_pages} ({elapsed:.1f}s, {len(md_local)} chars)", flush=True)
            return md_local

    step_start = time.perf_counter()
    page_dpi, render_profile = _choose_page_render_dpi(
        converter,
        speed_mode=speed_mode,
        page_index=page_index,
        is_references_page=is_references_page,
        image_names=image_names,
        visual_rects=visual_rects,
        base_dpi=dpi,
        page_class=page_class,
        adaptive_enabled=adaptive_page_budgets,
    )
    page_mat = mat
    if int(page_dpi) != int(dpi):
        try:
            zoom = float(page_dpi) / 72.0
            page_mat = fitz.Matrix(zoom, zoom)
        except Exception:
            page_mat = mat
    pix = page.get_pixmap(matrix=page_mat, alpha=False)
    png_bytes = pix.tobytes("png")
    png_bytes = _compress_png_bytes(png_bytes, speed_config=speed_config)
    _log_stage(4, "page render", step_start, f"bytes={len(png_bytes)} dpi={int(page_dpi)} profile={render_profile}")

    if metadata_rects:
        png_bytes = converter._mask_rects_on_png(
            png_bytes,
            rects=metadata_rects,
            page_width=page_w,
            page_height=page_h,
        )
        try:
            print(f"  [Page {page_index+1}] Masked {len(metadata_rects)} metadata region(s) before VL OCR", flush=True)
        except Exception:
            pass

    try:
        if bool(int(os.environ.get("KB_PDF_SAVE_PAGE_SCREENSHOTS", "0") or "0")):
            dbg_dir = assets_dir / "page_screenshots"
            dbg_dir.mkdir(exist_ok=True)
            (dbg_dir / f"page_{page_index+1}.png").write_bytes(png_bytes)
    except Exception:
        pass

    step_start = time.perf_counter()
    page_hint = _build_page_hint(
        converter,
        page_index=page_index,
        is_references_page=is_references_page,
        image_names=image_names,
        figure_meta_by_asset=figure_meta_by_asset,
    )
    layout_hint = _build_layout_page_hint(page=page, visual_rects=visual_rects)
    if layout_hint:
        page_hint = (page_hint + " " + layout_hint).strip() if page_hint else layout_hint
    png_bytes, page_hint, formula_placeholders = _apply_formula_overlay(
        converter,
        png_bytes=png_bytes,
        page=page,
        page_index=page_index,
        page_w=page_w,
        page_h=page_h,
        dpi=page_dpi,
        is_references_page=is_references_page,
        page_hint=page_hint,
        formula_candidates=formula_candidates,
    )
    _log_stage(
        5,
        "hints/overlay",
        step_start,
        f"formula={len(formula_placeholders)} hint={int(bool(page_hint))} density={plain_text_density} chars={plain_text_chars}",
    )

    max_tokens_override = _choose_page_max_tokens_override(
        speed_mode=speed_mode,
        page_index=page_index,
        is_references_page=is_references_page,
        page_hint=page_hint,
        image_names=image_names,
        visual_rects=visual_rects,
        formula_placeholders=formula_placeholders,
        plain_text_density=plain_text_density,
        page_class=page_class,
        adaptive_enabled=adaptive_page_budgets,
    )

    if log_stage_timings:
        print(
            f"[VISION_DIRECT][BUDGET] page {page_index+1}: "
            f"class={page_class} enabled={int(adaptive_page_budgets)} "
            f"dpi={int(page_dpi)} base_dpi={int(dpi)} "
            f"max_tokens={max_tokens_override if max_tokens_override is not None else 'default'} "
            f"density={plain_text_density} text_chars={plain_text_chars} "
            f"formulas={len(formula_candidates or [])} images={len(image_names)} visuals={len(visual_rects)} "
            f"profile={render_profile}",
            flush=True,
        )

    step_start = time.perf_counter()
    md = converter._convert_page_with_vision_guardrails(
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
        visual_rects=visual_rects,
        max_tokens_override=max_tokens_override,
        formula_placeholders=formula_placeholders,
        skip_references_column_mode=references_column_tried,
    )
    _log_stage(
        6,
        "vision convert",
        step_start,
        f"chars={len(md or '')} max_tokens={max_tokens_override if max_tokens_override is not None else 'default'}",
    )
    if not md:
        if str(speed_mode or "").strip().lower() == "ultra_fast":
            # Preserve the speed/quality contract at page granularity: most
            # pages use the fast 150-DPI path, while a page that fails every VL
            # and local fallback attempt gets one normal-quality rerender.
            print(
                f"[VISION_DIRECT] ultra-fast page {page_index+1} failed; retrying once with normal quality",
                flush=True,
            )
            normal_config = converter._get_speed_mode_config("normal", total_pages)
            try:
                normal_dpi = int(normal_config.get("dpi", 220) or 220)
            except Exception:
                normal_dpi = 220
            try:
                normal_mat = fitz.Matrix(normal_dpi / 72.0, normal_dpi / 72.0)
            except Exception:
                normal_mat = mat
            return process_vision_direct_page(
                converter,
                page=page,
                page_index=page_index,
                total_pages=total_pages,
                pdf_path=pdf_path,
                assets_dir=assets_dir,
                speed_mode="normal",
                speed_config=normal_config,
                dpi=normal_dpi,
                mat=normal_mat,
                started_at=t0,
            )
        print(f"[VISION_DIRECT] page {page_index+1} failed in both VL and fallback paths", flush=True)
        return None

    step_start = time.perf_counter()
    md = converter._postprocess_vision_page_markdown(
        md,
        page=page,
        page_index=page_index,
        pdf_path=pdf_path,
        assets_dir=assets_dir,
        image_names=image_names,
        figure_meta_by_asset=figure_meta_by_asset,
        visual_rects=visual_rects,
        is_references_page=is_references_page,
    )
    _log_stage(7, "postprocess", step_start, f"chars={len(md)}")
    total_elapsed = time.perf_counter() - perf_t0
    if log_stage_timings:
        print(f"  [Page {page_index+1}] TOTAL: {total_elapsed:.2f}s", flush=True)
    elapsed = time.time() - t0
    print(f"Finished page {page_index+1}/{total_pages} ({elapsed:.1f}s, {len(md)} chars)", flush=True)
    return md
