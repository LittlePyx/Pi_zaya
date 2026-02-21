from __future__ import annotations

import re
from typing import Optional, List
from pathlib import Path

try:
    import fitz
except ImportError:
    fitz = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

from .text_utils import _normalize_text
from .geometry_utils import _bbox_width, _rect_area, _rect_intersection_area, _overlap_1d


def _ensure_pdfplumber_module():
    global pdfplumber
    if pdfplumber is not None:
        return pdfplumber
    try:
        import pdfplumber as _pdfplumber
    except Exception as e:
        raise RuntimeError("`pdfplumber` package is not available.") from e
    pdfplumber = _pdfplumber
    return pdfplumber


def _escape_md_table_cell(value: str) -> str:
    cell = _normalize_text(value or "")
    if not cell:
        return ""
    cell = cell.replace("\r", "\n")
    cell = re.sub(r"\s*\n\s*", "<br>", cell)
    cell = cell.replace("|", r"\|")
    return cell.strip()


def _table_rows_to_markdown(rows_raw) -> Optional[str]:
    if not rows_raw or not isinstance(rows_raw, list):
        return None

    rows: list[list[str]] = []
    for row in rows_raw:
        if not isinstance(row, (list, tuple)):
            continue
        cells = [_escape_md_table_cell("" if c is None else str(c)) for c in row]
        rows.append(cells)

    # Drop empty rows.
    rows = [r for r in rows if any(c.strip() for c in r)]
    if len(rows) < 2:
        return None

    width = max(len(r) for r in rows)
    if width < 2:
        return None
    rows = [r + [""] * (width - len(r)) for r in rows]

    # Drop columns that are empty across all rows.
    keep_cols = [i for i in range(width) if any(rows[r][i].strip() for r in range(len(rows)))]
    if len(keep_cols) < 2:
        return None
    rows = [[r[i] for i in keep_cols] for r in rows]
    width = len(rows[0])

    # Some detectors prepend an almost-empty row; skip it if it looks like noise.
    if len(rows) >= 3:
        first_non_empty = sum(1 for c in rows[0] if c.strip())
        second_non_empty = sum(1 for c in rows[1] if c.strip())
        if first_non_empty <= 1 and second_non_empty >= 2:
            rows = rows[1:]

    if len(rows) < 2:
        return None

    header = rows[0]
    if not any(c.strip() for c in header):
        header = [f"col_{i + 1}" for i in range(width)]

    md_lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    for row in rows[1:]:
        md_lines.append("| " + " | ".join(row) + " |")
    return "\n".join(md_lines)


def _markdown_table_quality_score(md: str) -> float:
    lines = [ln.strip() for ln in (md or "").splitlines() if ln.strip()]
    if len(lines) < 3:
        return 0.0
    width = max(0, lines[0].count("|") - 1)
    body_rows = max(0, len(lines) - 2)
    non_empty_cells = 0
    for ln in lines:
        parts = [p.strip() for p in ln.strip("|").split("|")]
        non_empty_cells += sum(1 for p in parts if p and p != "---")
    return float(width * body_rows) + float(non_empty_cells) * 0.08


def _is_markdown_table_sane(md: str) -> bool:
    lines = [ln.strip() for ln in (md or "").splitlines() if ln.strip()]
    if len(lines) < 3:
        return False
    width = max(0, lines[0].count("|") - 1)
    if width < 2 or width > 18:
        return False
    cells: list[str] = []
    cols: list[list[str]] = [[] for _ in range(width)]
    numeric_cells = 0
    row_non_empty: list[int] = []
    filled_slots = 0
    total_slots = 0
    for ln in lines[2:]:
        parts = [p.strip() for p in ln.strip("|").split("|")]
        if len(parts) != width:
            return False
        non_empty_in_row = 0
        for ci, p in enumerate(parts):
            if p:
                non_empty_in_row += 1
                cols[ci].append(p)
                cells.append(p)
                if re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:%|e[+-]?\d+)?", p, flags=re.IGNORECASE):
                    numeric_cells += 1
        row_non_empty.append(non_empty_in_row)
        filled_slots += non_empty_in_row
        total_slots += width
    if not cells:
        return False
    if not row_non_empty:
        return False
    # Reject highly sparse grids (common false positives from figure axis text).
    fill_ratio = filled_slots / max(1, total_slots)
    if fill_ratio < 0.36:
        return False
    sparse_rows = sum(1 for n in row_non_empty if n <= max(1, width // 3))
    if sparse_rows / max(1, len(row_non_empty)) > 0.34:
        return False
    if numeric_cells == 0 and len(lines) >= 5:
        return False
    tiny_ratio = sum(1 for c in cells if len(c) <= 1) / max(1, len(cells))
    if tiny_ratio > 0.35:
        return False
    tiny_alpha_ratio = sum(1 for c in cells if re.fullmatch(r"[A-Za-z]{1,2}", c)) / max(1, len(cells))
    if tiny_alpha_ratio > 0.22:
        return False
    if len(lines) > 12 and tiny_alpha_ratio > 0.10:
        return False
    if width >= 5 and tiny_alpha_ratio > 0.14:
        return False
    for col_cells in cols:
        if len(col_cells) < 3:
            continue
        tiny_col = sum(1 for c in col_cells if len(c) <= 1) / max(1, len(col_cells))
        if tiny_col > 0.78:
            return False
        tiny_alpha_col = sum(1 for c in col_cells if re.fullmatch(r"[A-Za-z]{1,2}", c)) / max(1, len(col_cells))
        if tiny_alpha_col > 0.58:
            return False
    long_phrase_ratio = sum(1 for c in cells if len(re.findall(r"[A-Za-z]{2,}", c)) >= 8) / max(1, len(cells))
    if long_phrase_ratio > 0.55:
        return False
    if long_phrase_ratio > 0.38 and (numeric_cells / max(1, len(cells))) < 0.12:
        return False
    return True


def _extract_tables_by_pdfplumber(pdf_path: Optional[Path], page_index: int) -> list[tuple["fitz.Rect", str]]:
    if fitz is None or (pdf_path is None):
        return []
    try:
        pdm = _ensure_pdfplumber_module()
    except Exception:
        return []
    out: list[tuple[fitz.Rect, str]] = []
    try:
        with pdm.open(str(pdf_path)) as pd:
            if page_index < 0 or page_index >= len(pd.pages):
                return []
            pg = pd.pages[page_index]
            tables = pg.find_tables(
                table_settings={
                    "vertical_strategy": "text",
                    "horizontal_strategy": "text",
                    "snap_tolerance": 3,
                    "join_tolerance": 3,
                    "intersection_tolerance": 3,
                    "min_words_vertical": 2,
                    "min_words_horizontal": 1,
                }
            )
            for tb in tables:
                try:
                    bbox = tuple(float(x) for x in tb.bbox)
                    rect = fitz.Rect(bbox)
                except Exception:
                    continue
                try:
                    rows = tb.extract()
                except Exception:
                    rows = None
                md = _table_rows_to_markdown(rows) if rows is not None else None
                if not md:
                    continue
                if not _is_markdown_table_sane(md):
                    continue
                out.append((rect, md))
    except Exception:
        return []
    return out


def _page_maybe_has_table_from_dict(page_dict: dict) -> bool:
    """
    Fast page-level gate to avoid expensive table finder calls on obvious non-table pages.
    """
    blocks = page_dict.get("blocks", []) if isinstance(page_dict, dict) else []
    if not blocks:
        return False
    numeric_rows = 0
    delimiter_rows = 0
    scanned = 0
    for b in blocks:
        if "lines" not in b:
            continue
        for l in (b.get("lines", []) or []):
            spans = l.get("spans", []) or []
            if not spans:
                continue
            text = _normalize_text("".join(str(s.get("text", "")) for s in spans))
            if not text:
                continue
            scanned += 1
            if re.match(r"^\s*Table\s+(?:\d+|[IVXLC]+)\b", text, flags=re.IGNORECASE):
                return True
            if len(text) > 180:
                if scanned >= 260:
                    break
                continue
            cols = [c for c in re.split(r"\t+|\s{2,}", text.strip()) if c.strip()]
            nums = re.findall(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:%|e[+-]?\d+)?", text, flags=re.IGNORECASE)
            has_delim = ("|" in text) or (len(cols) >= 3)
            if has_delim:
                delimiter_rows += 1
            if len(nums) >= 2 and (has_delim or len(cols) >= 2):
                numeric_rows += 1

            if numeric_rows >= 3 and delimiter_rows >= 2:
                return True
            if delimiter_rows >= 6 and numeric_rows >= 2:
                return True
            if scanned >= 260:
                break
        if scanned >= 260:
            break
    return False

def table_text_to_markdown(text: str) -> Optional[str]:
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return None

    def split_cols(s: str) -> list[str]:
        return [c.strip() for c in re.split(r"\t+|\s{2,}", s.strip()) if c.strip()]

    def rows_to_md(rows: list[list[str]]) -> Optional[str]:
        rows = [r for r in rows if r and any(c.strip() for c in r)]
        if len(rows) < 2:
            return None
        width = max(len(r) for r in rows)
        if width < 2:
            return None
        rows = [r + [""] * (width - len(r)) for r in rows]
        # Drop columns empty across all rows.
        keep = [i for i in range(width) if any(rows[r][i].strip() for r in range(len(rows)))]
        if len(keep) < 2:
            return None
        rows = [[_escape_md_table_cell(r[i]) for i in keep] for r in rows]
        width = len(rows[0])
        header = rows[0]
        if not any(c.strip() for c in header):
            header = [f"col_{i + 1}" for i in range(width)]
        md_lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
        for r in rows[1:]:
            md_lines.append("| " + " | ".join(r) + " |")
        return "\n".join(md_lines)

    rows = [split_cols(ln) for ln in lines]
    rows = [r for r in rows if r]
    md_basic = rows_to_md(rows)
    if md_basic:
        return md_basic

    # Fallback: infer column boundaries from aligned whitespace runs (common in text-extracted tables).
    norm = [ln.replace("\t", "    ") for ln in lines]
    max_len = max(len(ln) for ln in norm)
    padded = [ln.ljust(max_len) for ln in norm]
    cut_votes: list[int] = []
    for ln in padded:
        for m in re.finditer(r"\s{2,}", ln):
            cut_votes.append(int((m.start() + m.end()) / 2))
    if not cut_votes:
        return None
    cut_votes.sort()
    clusters: list[list[int]] = []
    for pos in cut_votes:
        if not clusters or abs(pos - clusters[-1][-1]) > 2:
            clusters.append([pos])
        else:
            clusters[-1].append(pos)
    cuts = [int(round(sum(g) / len(g))) for g in clusters if len(g) >= max(2, len(padded) // 5)]
    cuts = sorted({c for c in cuts if 2 <= c <= max_len - 2})
    if not cuts:
        return None
    rows_aligned: list[list[str]] = []
    for ln in padded:
        start = 0
        row: list[str] = []
        for c in cuts + [max_len]:
            cell = ln[start:c].strip()
            row.append(cell)
            start = c
        if sum(1 for cell in row if cell.strip()) >= 2:
            rows_aligned.append(row)
    return rows_to_md(rows_aligned)


def _table_from_numeric_pattern(lines: list[str]) -> Optional[str]:
    """
    Recover simple text tables like:
      Method PSNR SSIM LPIPS
      Ours 33.8 0.95 0.08
    where spacing is collapsed and first column may contain words.
    """
    if len(lines) < 3:
        return None
    num_re = re.compile(r"^[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:%|e[+-]?\d+)?$", re.IGNORECASE)
    token_rows = [re.findall(r"\S+", ln.strip()) for ln in lines if ln.strip()]
    if len(token_rows) < 3:
        return None

    num_counts = [sum(1 for t in row if num_re.fullmatch(t)) for row in token_rows]
    data_counts = [c for c in num_counts if c >= 2]
    if len(data_counts) < 2:
        return None
    # Robust central tendency without importing statistics.
    data_counts.sort()
    n_num = data_counts[len(data_counts) // 2]
    if n_num < 2:
        return None

    rows: list[list[str]] = []
    for row in token_rows:
        if len(row) < n_num + 1:
            continue
        first_num_idx = next((i for i, t in enumerate(row) if num_re.fullmatch(t)), None)
        if first_num_idx is None:
            # header row: use last n_num tokens as metric columns
            label = " ".join(row[: len(row) - n_num]).strip()
            vals = row[len(row) - n_num :]
            if (not label) or len(vals) != n_num:
                continue
            rows.append([label] + vals)
            continue
        nums = [t for t in row[first_num_idx:] if num_re.fullmatch(t)]
        if len(nums) < n_num:
            continue
        label = " ".join(row[:first_num_idx]).strip()
        if not label:
            continue
        rows.append([label] + nums[:n_num])

    if len(rows) < 3:
        return None
    width = 1 + n_num
    rows = [r + [""] * (width - len(r)) for r in rows]
    header = [_escape_md_table_cell(c) for c in rows[0]]
    md_lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    for r in rows[1:]:
        md_lines.append("| " + " | ".join(_escape_md_table_cell(c) for c in r) + " |")
    out = "\n".join(md_lines)
    if not _is_markdown_table_sane(out):
        return None
    return out

def _extract_tables_by_layout(
    page,
    *,
    pdf_path: Optional[Path] = None,
    page_index: int = 0,
    visual_rects: Optional[list["fitz.Rect"]] = None,
    use_pdfplumber_fallback: bool = False,
) -> list[tuple["fitz.Rect", str]]:
    if fitz is None:
        return []

    page_w = float(page.rect.width)
    page_h = float(page.rect.height)
    page_area = max(1.0, page_w * page_h)
    vis_rects = visual_rects or []
    has_table_hint = getattr(page, "has_table_hint", False)

    # Heuristic: detect captions like "Table 1:" to anchor the table search.
    caption_rects: list[fitz.Rect] = []
    try:
        d = page.get_text("dict")
        for b in d.get("blocks", []):
            if "lines" not in b:
                continue
            bbox = b.get("bbox")
            lines: list[str] = []
            for l in b.get("lines", []):
                spans = l.get("spans", [])
                if not spans:
                    continue
                line = "".join(str(s.get("text", "")) for s in spans)
                line = _normalize_text(line)
                if line:
                    lines.append(line)
            txt = _normalize_text(" ".join(lines))
            if re.match(r"^\s*Table\s+(?:\d+|[IVXLC]+)\b", txt, flags=re.IGNORECASE):
                caption_rects.append(fitz.Rect(tuple(float(x) for x in bbox)))
    except Exception:
        caption_rects = []

    def _has_nearby_table_caption(rect: "fitz.Rect") -> bool:
        if not caption_rects:
            return False
        for cr in caption_rects:
            hov = _overlap_1d(float(rect.x0), float(rect.x1), float(cr.x0) - 18.0, float(cr.x1) + 18.0)
            min_hov = max(10.0, min(float(rect.width), float(cr.width) + 36.0) * 0.15)
            if hov < min_hov:
                continue
            vgap = min(abs(float(rect.y0) - float(cr.y1)), abs(float(cr.y0) - float(rect.y1)))
            if vgap <= max(110.0, page_h * 0.24):
                return True
        return False

    primary_kwargs = [{"vertical_strategy": "lines", "horizontal_strategy": "lines"}]
    if has_table_hint:
        primary_kwargs.extend(
            [
                {"vertical_strategy": "lines", "horizontal_strategy": "text", "min_words_horizontal": 1, "text_tolerance": 2.0},
                {"vertical_strategy": "text", "horizontal_strategy": "lines", "min_words_vertical": 2, "text_tolerance": 2.0},
            ]
        )

    candidates: list[tuple[fitz.Rect, str, float]] = []
    kwargs_seq = primary_kwargs
    
    import time
    table_extract_start = time.time()
    max_table_extract_time = 8.0  # Max 8 seconds for table extraction per page
    
    for strategy_idx, kwargs in enumerate(kwargs_seq):
        if time.time() - table_extract_start > max_table_extract_time:
            print(f"      [Table extraction] Timeout after {strategy_idx}/{len(kwargs_seq)} strategies, skipping remaining", flush=True)
            break
        
        uses_text_strategy = ("text" in str(kwargs.get("vertical_strategy", "")).lower()) or (
            "text" in str(kwargs.get("horizontal_strategy", "")).lower()
        )
        strategy_start = time.time()
        try:
            table_finder = page.find_tables(**kwargs)
            strategy_time = time.time() - strategy_start
            if strategy_time > 1.0:
                print(f"      [Table extraction] Strategy {strategy_idx+1} ({kwargs.get('vertical_strategy', '?')}/{kwargs.get('horizontal_strategy', '?')}): {strategy_time:.2f}s (SLOW!)", flush=True)
        except Exception as e:
            strategy_time = time.time() - strategy_start
            if strategy_time > 0.5:
                print(f"      [Table extraction] Strategy {strategy_idx+1} FAILED after {strategy_time:.2f}s: {e}", flush=True)
            continue
        tables = getattr(table_finder, "tables", table_finder)
        if not tables:
            continue

        for tb in tables:
            try:
                rect = fitz.Rect(getattr(tb, "bbox", None))
            except Exception:
                continue
            if _rect_area(rect) < page_area * 0.0035:
                continue
            if float(rect.width) < page_w * 0.12 or float(rect.height) < page_h * 0.04:
                continue
            if _rect_area(rect) > page_area * 0.55:
                continue

            md = None
            try:
                md = _table_rows_to_markdown(tb.extract())
            except Exception:
                md = None
            if not md:
                try:
                    raw_clip = page.get_text("text", clip=rect)
                except Exception:
                    raw_clip = ""
                md = table_text_to_markdown(raw_clip) if raw_clip else None
                if (not md) and raw_clip:
                    md = _table_from_numeric_pattern([ln for ln in raw_clip.splitlines() if ln.strip()])
            if not md:
                continue
            if not _is_markdown_table_sane(md):
                continue
            near_caption = _has_nearby_table_caption(rect)
            if vis_rects and (not near_caption):
                vis_overlap = max(
                    (
                        _rect_intersection_area(rect, vr) / max(1.0, min(_rect_area(rect), _rect_area(vr)))
                        for vr in vis_rects
                    ),
                    default=0.0,
                )
                # Strongly suppress figure-overlapping table false positives.
                if vis_overlap >= 0.62:
                    continue
                if vis_overlap >= 0.45 and (not has_table_hint):
                    continue
            if uses_text_strategy and (not near_caption):
                # Text-based strategies are prone to chart-axis false positives.
                # Keep only compact, denser candidates when no table caption is nearby.
                if _rect_area(rect) > page_area * 0.18:
                    continue
                if len([ln for ln in md.splitlines() if ln.strip()]) < 4:
                    continue
            if (not has_table_hint) and (not near_caption) and _rect_area(rect) > page_area * 0.08:
                continue
            score = _markdown_table_quality_score(md)
            if score <= 0.0:
                continue
            candidates.append((rect, md, score))

    if (not candidates) and use_pdfplumber_fallback and (pdf_path is not None) and has_table_hint:
        for rect, md in _extract_tables_by_pdfplumber(pdf_path, page_index):
            score = _markdown_table_quality_score(md)
            if score > 0.0:
                candidates.append((rect, md, score))

    if not candidates:
        return []

    # De-duplicate overlapping candidates, keeping the better table representation.
    uniq: list[tuple[fitz.Rect, str, float]] = []
    for rect, md, score in sorted(candidates, key=lambda x: (x[0].y0, x[0].x0, -x[2])):
        replaced = False
        for i, (r0, md0, s0) in enumerate(uniq):
            inter = _rect_intersection_area(rect, r0)
            denom = max(1.0, min(_rect_area(rect), _rect_area(r0)))
            if inter / denom >= 0.72:
                if score > s0:
                    uniq[i] = (rect, md, score)
                replaced = True
                break
        if not replaced:
            uniq.append((rect, md, score))

    uniq.sort(key=lambda x: (x[0].y0, x[0].x0))
    return [(r, md) for r, md, _ in uniq]


def _latex_array_to_markdown_table(latex: str) -> Optional[str]:
    # Accept \begin{array}{|l|c|...|} ... \end{array} or tabular-like.
    m = re.search(r"\\begin\{array\}\{[^}]*\}(.*?)\\end\{array\}", latex, flags=re.S)
    if not m:
        return None
    body = m.group(1)
    body = body.replace("\\hline", "")
    body = body.strip()
    # split rows
    raw_rows = [r.strip() for r in body.split("\\\\") if r.strip()]
    if len(raw_rows) < 2:
        return None
    rows: list[list[str]] = []
    for r in raw_rows:
        cols = [c.strip() for c in r.split("&")]
        cols = [re.sub(r"\\text\{([^}]*)\}", r"\1", c) for c in cols]
        cols = [re.sub(r"\s+", " ", c).strip() for c in cols]
        rows.append(cols)
    width = max(len(r) for r in rows)
    if width < 2:
        return None
    rows = [r + [""] * (width - len(r)) for r in rows]
    header = rows[0]
    md_lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    for r in rows[1:]:
        md_lines.append("| " + " | ".join(r) + " |")
    return "\n".join(md_lines)

def _convert_latex_array_math_blocks(md: str) -> str:
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        if lines[i].strip() != "$$":
            out.append(lines[i])
            i += 1
            continue
        j = i + 1
        buf: list[str] = []
        while j < len(lines) and lines[j].strip() != "$$":
            buf.append(lines[j])
            j += 1
        if j >= len(lines):
            out.append(lines[i])
            out.extend(buf)
            break
        latex = "\n".join(buf)
        table = _latex_array_to_markdown_table(latex)
        if table:
            out.extend(table.splitlines())
            out.append("")
        else:
            out.append("$$")
            out.extend(buf)
            out.append("$$")
        i = j + 1
    return "\n".join(out)
