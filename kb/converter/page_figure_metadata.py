from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Optional

try:
    import fitz
except ImportError:
    fitz = None

from .geometry_utils import _overlap_1d
from .text_utils import _normalize_text


def extract_page_figure_caption_candidates(page) -> list[dict]:
    if fitz is None or page is None:
        return []
    out: list[dict] = []
    try:
        d = page.get_text("dict") or {}
    except Exception:
        return out

    cap_start_re = re.compile(
        r"^\s*(?:fig(?:ure)?\.?)\s*(\d{1,4}[A-Za-z]?)\b",
        flags=re.IGNORECASE,
    )
    for b in d.get("blocks", []) or []:
        if "lines" not in b:
            continue
        bbox = b.get("bbox")
        if not bbox:
            continue
        txt_parts: list[str] = []
        for l in b.get("lines", []) or []:
            spans = l.get("spans", []) or []
            if not spans:
                continue
            line = "".join(str(s.get("text", "")) for s in spans)
            line = _normalize_text(line).strip()
            if line:
                txt_parts.append(line)
        if not txt_parts:
            continue
        txt = _normalize_text(" ".join(txt_parts)).strip()
        if not txt:
            continue
        m = cap_start_re.match(txt)
        if not m:
            continue
        fig_ident = (m.group(1) or "").strip()
        fig_no = None
        try:
            fig_no = int(re.match(r"^(\d+)", fig_ident).group(1))  # type: ignore[union-attr]
        except Exception:
            fig_no = None
        out.append(
            {
                "fig_no": fig_no,
                "fig_ident": fig_ident,
                "caption": txt,
                "bbox": [float(x) for x in bbox],
            }
        )
    out.sort(key=lambda x: (float(x["bbox"][1]), float(x["bbox"][0])))
    return out


def match_figure_entries_with_captions(
    *,
    page,
    figure_entries: list[dict],
    caption_candidates: list[dict],
) -> list[dict]:
    if fitz is None or page is None:
        return figure_entries
    if not figure_entries or not caption_candidates:
        return figure_entries

    page_h = float(page.rect.height or 0.0)
    max_below_gap = max(120.0, page_h * 0.34)
    max_above_gap = max(96.0, page_h * 0.22)

    def _score(entry: dict, cap: dict) -> float:
        rb = entry.get("crop_bbox") or entry.get("bbox")
        cb = cap.get("bbox")
        if not rb or not cb:
            return -10_000.0
        try:
            rx0, ry0, rx1, ry1 = [float(v) for v in rb]
            cx0, cy0, cx1, cy1 = [float(v) for v in cb]
        except Exception:
            return -10_000.0

        hov = _overlap_1d(rx0, rx1, cx0 - 18.0, cx1 + 18.0)
        min_w = max(1.0, min(rx1 - rx0, (cx1 - cx0) + 36.0))
        hov_ratio = hov / min_w

        below_gap = cy0 - ry1
        above_gap = ry0 - cy1
        gap_penalty = 500.0
        if below_gap >= -20.0 and below_gap <= max_below_gap:
            gap_penalty = max(0.0, below_gap) * 0.9
        elif above_gap >= -20.0 and above_gap <= max_above_gap:
            gap_penalty = 60.0 + max(0.0, above_gap) * 1.2

        rcx = (rx0 + rx1) / 2.0
        ccx = (cx0 + cx1) / 2.0
        x_penalty = abs(rcx - ccx) * 0.06

        score = (hov_ratio * 140.0) - gap_penalty - x_penalty
        if hov_ratio < 0.08:
            score -= 140.0
        return score

    used_entries: set[int] = set()
    assigned: dict[int, dict] = {}
    for cap in caption_candidates:
        best_idx = None
        best_score = -10_000.0
        for i, ent in enumerate(figure_entries):
            if i in used_entries:
                continue
            s = _score(ent, cap)
            if s > best_score:
                best_score = s
                best_idx = i
        if best_idx is None:
            continue
        if best_score < -18.0:
            continue
        used_entries.add(best_idx)
        assigned[best_idx] = cap

    out: list[dict] = []
    for i, ent in enumerate(figure_entries):
        row = dict(ent)
        cap = assigned.get(i)
        if cap:
            row["fig_no"] = cap.get("fig_no")
            row["fig_ident"] = cap.get("fig_ident")
            row["caption"] = cap.get("caption")
            row["caption_bbox"] = cap.get("bbox")
        out.append(row)
    return out


def split_visual_rects_by_internal_captions(
    *,
    page,
    visual_rects: list["fitz.Rect"],
    caption_candidates: list[dict],
) -> list["fitz.Rect"]:
    if fitz is None or page is None:
        return visual_rects
    if not visual_rects or not caption_candidates:
        return visual_rects

    page_h = float(page.rect.height or 0.0)
    min_segment_height = max(72.0, page_h * 0.10)
    split_gap = max(4.0, page_h * 0.004)
    out: list["fitz.Rect"] = []

    for rect in visual_rects:
        r = fitz.Rect(rect)
        if r.width <= 0.0 or r.height <= 0.0:
            continue
        internal_caps: list["fitz.Rect"] = []
        for cap in caption_candidates:
            try:
                cr = fitz.Rect(cap.get("bbox"))
            except Exception:
                continue
            if cr.width <= 0.0 or cr.height <= 0.0:
                continue
            x_ov = _overlap_1d(float(r.x0), float(r.x1), float(cr.x0) - 16.0, float(cr.x1) + 16.0)
            min_w = max(1.0, min(float(r.width), float(cr.width) + 32.0))
            if (x_ov / min_w) < 0.42:
                continue
            above_space = float(cr.y0) - float(r.y0)
            below_space = float(r.y1) - float(cr.y1)
            if above_space < min_segment_height:
                continue
            if below_space < min_segment_height:
                continue
            internal_caps.append(cr)

        if not internal_caps:
            out.append(r)
            continue

        internal_caps.sort(key=lambda rr: (float(rr.y0), float(rr.x0)))
        start_y = float(r.y0)
        made_split = False
        for cr in internal_caps:
            seg_end = max(start_y + 1.0, float(cr.y0) - split_gap)
            if (seg_end - start_y) >= min_segment_height:
                out.append(fitz.Rect(float(r.x0), start_y, float(r.x1), seg_end))
                made_split = True
            start_y = min(float(r.y1), float(cr.y1) + split_gap)
        if (float(r.y1) - start_y) >= min_segment_height:
            out.append(fitz.Rect(float(r.x0), start_y, float(r.x1), float(r.y1)))
            made_split = True
        if not made_split:
            out.append(r)

    return out


def persist_page_figure_metadata(
    *,
    assets_dir: Path,
    page_index: int,
    figure_entries: list[dict],
) -> dict[str, dict]:
    """
    Persist per-page figure metadata (asset/crop/caption binding) and return
    a fast lookup map by asset filename.
    """
    page_no = int(page_index) + 1
    if not figure_entries:
        meta_path = assets_dir / f"page_{page_no}_fig_index.json"
        try:
            meta_path.unlink(missing_ok=True)
        except Exception:
            pass
        rows = _rebuild_document_figure_identity(assets_dir)
        return {
            str(row.get("asset_name") or ""): row
            for row in rows
            if int(row.get("page") or 0) == page_no and str(row.get("asset_name") or "").strip()
        }

    rows: list[dict] = []
    for idx, item in enumerate(figure_entries, start=1):
        asset_name = str(item.get("asset_name") or "").strip()
        if not asset_name:
            continue
        row = {
            "page": page_no,
            "index": int(idx),
            "asset_name": asset_name,
            "fig_no": item.get("fig_no"),
            "fig_ident": item.get("fig_ident"),
            "caption": item.get("caption"),
            "bbox": item.get("bbox"),
            "crop_bbox": item.get("crop_bbox"),
            "caption_bbox": item.get("caption_bbox"),
        }
        rows.append(row)

    meta_path = assets_dir / f"page_{page_no}_fig_index.json"
    try:
        payload = {
            "page": page_no,
            "figures": rows,
        }
        meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    enriched_rows = _rebuild_document_figure_identity(assets_dir)
    return {
        str(row.get("asset_name") or ""): row
        for row in enriched_rows
        if int(row.get("page") or 0) == page_no and str(row.get("asset_name") or "").strip()
    }


def build_page_figure_mapping_hint(figure_meta_by_asset: Optional[dict[str, dict]]) -> str:
    if not figure_meta_by_asset:
        return ""
    rows: list[tuple[int, str]] = []
    for asset, meta in figure_meta_by_asset.items():
        if not isinstance(meta, dict):
            continue
        try:
            n = int(meta.get("fig_no"))
        except Exception:
            continue
        rows.append((n, str(asset)))
    if not rows:
        return ""
    rows.sort(key=lambda x: (x[0], x[1].lower()))
    seen: set[int] = set()
    pairs: list[str] = []
    for n, asset in rows:
        if n in seen:
            continue
        seen.add(n)
        pairs.append(f"Figure {n} -> ./assets/{asset}")
    if not pairs:
        return ""
    return "Figure-to-asset mapping for this page: " + "; ".join(pairs) + "."


_FIG_ALIAS_RE = re.compile(r"^fig_\d+\.[A-Za-z0-9]+$", flags=re.IGNORECASE)
_MD_IMG_RE = re.compile(r"^\s*!\[[^\]]*\]\(\./assets/([^)]+)\)\s*$", flags=re.IGNORECASE)
_MD_HEADING_RE = re.compile(r"^\s*#{1,6}\s+")


def _int_or_zero(value) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _binding_source(row: dict) -> str:
    if _int_or_zero(row.get("fig_no")) > 0 and str(row.get("caption") or "").strip():
        return "caption_match"
    if _int_or_zero(row.get("fig_no")) > 0:
        return "figure_number_only"
    return "visual_order"


def _binding_confidence(row: dict) -> float:
    if _int_or_zero(row.get("fig_no")) > 0 and str(row.get("caption") or "").strip():
        return 0.98
    if _int_or_zero(row.get("fig_no")) > 0:
        return 0.72
    return 0.28


def _load_all_page_rows(assets_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for meta_path in sorted(assets_dir.glob("page_*_fig_index.json")):
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        figures = payload.get("figures") if isinstance(payload, dict) else None
        if not isinstance(figures, list):
            continue
        for idx, item in enumerate(figures, start=1):
            if not isinstance(item, dict):
                continue
            row = dict(item)
            row["page"] = _int_or_zero(row.get("page") or payload.get("page"))
            row["index"] = _int_or_zero(row.get("index") or idx)
            asset_name = str(row.get("asset_name") or "").strip()
            if not asset_name:
                continue
            row["asset_name"] = asset_name
            rows.append(row)
    rows.sort(key=lambda item: (_int_or_zero(item.get("page")), _int_or_zero(item.get("index")), str(item.get("asset_name") or "").lower()))
    return rows


def _canonicalize_figure_rows(rows: list[dict]) -> list[dict]:
    figure_counts: dict[int, int] = {}
    for row in rows:
        fig_no = _int_or_zero(row.get("fig_no"))
        if fig_no > 0:
            figure_counts[fig_no] = int(figure_counts.get(fig_no) or 0) + 1

    out: list[dict] = []
    for row in rows:
        rec = dict(row)
        page_no = max(1, _int_or_zero(rec.get("page")))
        item_idx = max(1, _int_or_zero(rec.get("index")))
        fig_no = _int_or_zero(rec.get("fig_no"))
        fig_ident = str(rec.get("fig_ident") or "").strip()
        asset_name = str(rec.get("asset_name") or "").strip()
        suffix = Path(asset_name).suffix or ".png"
        unique_number = fig_no > 0 and int(figure_counts.get(fig_no) or 0) == 1

        if fig_no > 0 and unique_number:
            figure_id = f"fig_{fig_no:03d}"
            asset_alias = f"fig_{fig_no}{suffix.lower()}"
        elif fig_no > 0:
            figure_id = f"fig_{fig_no:03d}_p{page_no:03d}_i{item_idx:02d}"
            asset_alias = ""
        else:
            figure_id = f"page_{page_no:03d}_fig_{item_idx:02d}"
            asset_alias = ""

        rec["page"] = page_no
        rec["index"] = item_idx
        rec["paper_figure_number"] = fig_no if fig_no > 0 else None
        rec["figure_ident"] = fig_ident or (str(fig_no) if fig_no > 0 else "")
        rec["figure_id"] = figure_id
        rec["asset_name_raw"] = asset_name
        rec["asset_name_alias"] = asset_alias
        rec["binding_source"] = _binding_source(rec)
        rec["binding_confidence"] = _binding_confidence(rec)
        out.append(rec)
    return out


def _write_sidecar(assets_dir: Path, row: dict) -> None:
    asset_name = str(row.get("asset_name") or "").strip()
    if not asset_name:
        return
    sidecar = assets_dir / f"{Path(asset_name).stem}.meta.json"
    try:
        sidecar.write_text(
            json.dumps(row, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def _write_page_indices(assets_dir: Path, rows: list[dict]) -> None:
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        page_no = max(1, _int_or_zero(row.get("page")))
        grouped.setdefault(page_no, []).append(dict(row))
        _write_sidecar(assets_dir, row)

    for page_no, items in grouped.items():
        items.sort(key=lambda item: (_int_or_zero(item.get("index")), str(item.get("asset_name") or "").lower()))
        meta_path = assets_dir / f"page_{page_no}_fig_index.json"
        payload = {"page": int(page_no), "figures": items}
        try:
            meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass


def _write_document_figure_index(assets_dir: Path, rows: list[dict]) -> None:
    doc_index = assets_dir / "figure_index.json"
    if not rows:
        try:
            doc_index.unlink(missing_ok=True)
        except Exception:
            pass
        return
    payload = {
        "figures": rows,
    }
    try:
        doc_index.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _sync_alias_files(assets_dir: Path, rows: list[dict]) -> None:
    desired: dict[str, str] = {}
    for row in rows:
        alias_name = str(row.get("asset_name_alias") or "").strip()
        raw_name = str(row.get("asset_name_raw") or row.get("asset_name") or "").strip()
        if not alias_name or not raw_name:
            continue
        raw_path = assets_dir / raw_name
        if not raw_path.exists():
            continue
        desired[alias_name] = raw_name

    for p in assets_dir.glob("fig_*.*"):
        if not p.is_file():
            continue
        if not _FIG_ALIAS_RE.match(p.name):
            continue
        if p.name not in desired:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass

    for alias_name, raw_name in desired.items():
        raw_path = assets_dir / raw_name
        alias_path = assets_dir / alias_name
        try:
            if alias_path.resolve(strict=False) == raw_path.resolve(strict=False):
                continue
        except Exception:
            pass
        try:
            shutil.copyfile(raw_path, alias_path)
        except Exception:
            pass


def _rebuild_document_figure_identity(assets_dir: Path) -> list[dict]:
    rows = _canonicalize_figure_rows(_load_all_page_rows(assets_dir))
    if not rows:
        _write_document_figure_index(assets_dir, [])
        _sync_alias_files(assets_dir, [])
        return []
    _write_page_indices(assets_dir, rows)
    _write_document_figure_index(assets_dir, rows)
    _sync_alias_files(assets_dir, rows)
    return rows


def _parse_markdown_figure_caption(line: str) -> Optional[dict]:
    st = _normalize_text(str(line or "")).strip()
    if not st:
        return None
    st = re.sub(r"^\*{1,2}\s*", "", st)
    st = re.sub(r"\s*\*{1,2}\s*$", "", st)
    st = st.replace("**", "").replace("*", "").replace("`", "")
    match = re.match(r"^(?:Figure|Fig\.?)\s*(\d{1,4}[A-Za-z]?)\s*[.:]?\s*(.*)$", st, flags=re.IGNORECASE)
    if not match:
        return None
    ident = str(match.group(1) or "").strip()
    tail = str(match.group(2) or "").strip()
    fig_no = None
    try:
        fig_no = int(re.match(r"^(\d+)", ident).group(1))  # type: ignore[union-attr]
    except Exception:
        fig_no = None
    caption = f"Figure {ident}."
    if tail:
        caption = f"{caption} {tail}"
    return {
        "fig_no": fig_no,
        "fig_ident": ident,
        "caption": caption.strip(),
    }


def _find_markdown_caption_for_image(lines: list[str], line_index: int) -> Optional[dict]:
    def _scan(*, delta: int, max_steps: int, max_non_empty: int) -> Optional[dict]:
        non_empty = 0
        for step in range(1, max_steps + 1):
            idx = line_index + delta * step
            if not (0 <= idx < len(lines)):
                break
            raw = lines[idx] or ""
            stripped = raw.strip()
            if not stripped:
                continue
            if _MD_HEADING_RE.match(raw) or _MD_IMG_RE.match(raw):
                break
            non_empty += 1
            parsed = _parse_markdown_figure_caption(stripped)
            if parsed:
                return parsed
            if non_empty >= max_non_empty:
                break
        return None

    return _scan(delta=1, max_steps=10, max_non_empty=3) or _scan(delta=-1, max_steps=6, max_non_empty=2)


def reconcile_figure_metadata_from_markdown(*, md: str, assets_dir: Path) -> dict[str, dict]:
    if not md:
        return {}

    rows = _canonicalize_figure_rows(_load_all_page_rows(assets_dir))
    if not rows:
        return {}

    row_by_asset_key: dict[str, int] = {}
    for idx, row in enumerate(rows):
        for key in (
            str(row.get("asset_name") or "").strip(),
            str(row.get("asset_name_raw") or "").strip(),
            str(row.get("asset_name_alias") or "").strip(),
        ):
            if key:
                row_by_asset_key[key.lower()] = idx

    lines = md.splitlines()
    changed = False
    for line_index, raw in enumerate(lines):
        match = _MD_IMG_RE.match(raw or "")
        if not match:
            continue
        asset_ref = Path(str(match.group(1) or "").strip()).name.lower()
        row_idx = row_by_asset_key.get(asset_ref)
        if row_idx is None:
            continue
        parsed = _find_markdown_caption_for_image(lines, line_index)
        if not parsed:
            continue
        row = dict(rows[row_idx])
        if parsed.get("caption") and str(row.get("caption") or "").strip() != str(parsed.get("caption") or "").strip():
            row["caption"] = parsed.get("caption")
            changed = True
        if parsed.get("fig_ident") and str(row.get("fig_ident") or "").strip() != str(parsed.get("fig_ident") or "").strip():
            row["fig_ident"] = parsed.get("fig_ident")
            changed = True
        if parsed.get("fig_no") and _int_or_zero(row.get("fig_no")) != int(parsed.get("fig_no") or 0):
            row["fig_no"] = int(parsed.get("fig_no") or 0)
            changed = True
        rows[row_idx] = row

    if not changed:
        return {
            str(row.get("asset_name") or ""): row
            for row in _canonicalize_figure_rows(rows)
            if str(row.get("asset_name") or "").strip()
        }

    updated_rows = _canonicalize_figure_rows(rows)
    _write_page_indices(assets_dir, updated_rows)
    _write_document_figure_index(assets_dir, updated_rows)
    _sync_alias_files(assets_dir, updated_rows)
    return {
        str(row.get("asset_name") or ""): row
        for row in updated_rows
        if str(row.get("asset_name") or "").strip()
    }
