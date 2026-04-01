from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from .text_utils import _normalize_text


def inject_missing_page_image_links(
    md: str,
    *,
    page_index: int,
    image_names: list[str],
    figure_meta_by_asset: Optional[dict[str, dict]] = None,
    is_references_page: bool = False,
) -> str:
    """
    In vision-direct mode, VL output can omit image markdown even when page images
    were extracted to ./assets. Inject deterministic links as a fallback.
    """
    if not md:
        return md
    if is_references_page:
        return md
    if not image_names:
        return md

    def _img_key(nm: str) -> tuple[int, str]:
        m = re.search(r"_fig_(\d+)", nm, flags=re.IGNORECASE)
        if m:
            try:
                return (int(m.group(1)), nm.lower())
            except Exception:
                pass
        return (10**9, nm.lower())

    ordered = sorted({str(n).strip() for n in image_names if str(n).strip()}, key=_img_key)
    if not ordered:
        return md

    link_re = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
    existing_basenames: set[str] = set()
    for m in link_re.finditer(md):
        raw = (m.group(1) or "").strip().strip('"').strip("'")
        if not raw:
            continue
        existing_basenames.add(Path(raw).name)

    missing = [nm for nm in ordered if nm not in existing_basenames]
    if not missing:
        return md

    lines = md.splitlines()
    cap_line_re = re.compile(
        r"^\s*(?:[*_`>#\[\(]\s*)*(?:Figure|Fig\.?)\s*(\d+)\b",
        flags=re.IGNORECASE,
    )
    cap_idx: list[int] = []
    cap_lines_by_fig: dict[int, list[int]] = {}
    cap_fig_no: str | None = None
    for i, ln in enumerate(lines):
        st = _normalize_text(ln or "").strip()
        m = cap_line_re.match(st)
        if m:
            cap_idx.append(i)
            try:
                fig_no = int(m.group(1))
                cap_lines_by_fig.setdefault(fig_no, []).append(i)
            except Exception:
                pass
            if cap_fig_no is None:
                cap_fig_no = m.group(1)

    looks_figure_page = bool(cap_idx) or bool(re.search(r"\bFigure\s+\d+\b", md, flags=re.IGNORECASE))
    if not looks_figure_page:
        return md

    figure_meta_by_asset = figure_meta_by_asset or {}
    pending: list[tuple[int, str]] = []
    for nm in missing:
        meta = figure_meta_by_asset.get(str(nm)) if isinstance(figure_meta_by_asset, dict) else None
        fig_no = None
        if isinstance(meta, dict):
            try:
                fig_no = int(meta.get("fig_no"))
            except Exception:
                fig_no = None
        order_key = 10**9 if fig_no is None else fig_no
        pending.append((order_key, str(nm)))
    pending.sort(key=lambda x: (x[0], x[1].lower()))

    inserts_by_line: dict[int, list[str]] = {}
    remaining: list[str] = []
    for _, nm in pending:
        meta = figure_meta_by_asset.get(str(nm)) if isinstance(figure_meta_by_asset, dict) else None
        fig_no = None
        if isinstance(meta, dict):
            try:
                fig_no = int(meta.get("fig_no"))
            except Exception:
                fig_no = None
        if fig_no is None:
            remaining.append(nm)
            continue
        cap_lines = cap_lines_by_fig.get(fig_no) or []
        if not cap_lines:
            remaining.append(nm)
            continue
        insert_at = cap_lines[0]
        alt = f"Figure {fig_no}"
        inserts_by_line.setdefault(insert_at, []).append(f"![{alt}](./assets/{nm})")

    if inserts_by_line:
        rebuilt: list[str] = []
        for idx, ln in enumerate(lines):
            blocks = inserts_by_line.get(idx) or []
            if blocks:
                for ref in blocks:
                    rebuilt.append(ref)
                    rebuilt.append("")
            rebuilt.append(ln)
        lines = rebuilt

    if remaining:
        insert_at = cap_idx[0] if cap_idx else 0
        alt = f"Figure {cap_fig_no}" if cap_fig_no else "Figure"
        inject_block: list[str] = [f"![{alt}](./assets/{nm})" for nm in remaining]
        if inject_block:
            inject_block.append("")
            lines = lines[:insert_at] + inject_block + lines[insert_at:]
    if inserts_by_line or remaining:
        try:
            print(
                f"[IMAGE_FIX] page {page_index+1}: injected {len(missing)} missing image link(s)",
                flush=True,
            )
        except Exception:
            pass
    return "\n".join(lines)


def inject_page_image_captions_from_meta(
    md: str,
    *,
    page_index: int,
    figure_meta_by_asset: Optional[dict[str, dict]] = None,
) -> str:
    if not md or not figure_meta_by_asset:
        return md

    page_no = int(page_index) + 1
    img_re = re.compile(
        rf"^\s*!\[[^\]]*\]\(\./assets/(page_{page_no}_fig_\d+\.[^)]+)\)\s*$",
        flags=re.IGNORECASE,
    )
    heading_re = re.compile(r"^\s*#{1,6}\s+")
    cap_re = re.compile(
        r"^\s*(?:[*_`>#\[\(]\s*)*(?:fig(?:ure)?\.?)\s*(\d{1,4})(?:[A-Za-z])?\b",
        flags=re.IGNORECASE,
    )

    def _canonical_caption(meta: dict) -> Optional[str]:
        raw = _normalize_text(str(meta.get("caption") or "")).strip()
        if not raw:
            return None
        m = re.match(r"^\s*fig(?:ure)?\.?\s*(\d{1,4}[A-Za-z]?)\s*[.:]?\s*(.*)$", raw, flags=re.IGNORECASE)
        if not m:
            return raw
        ident = (m.group(1) or "").strip()
        tail = (m.group(2) or "").strip()
        if tail:
            return f"Figure {ident}. {tail}"
        return f"Figure {ident}."

    lines = md.splitlines()
    out: list[str] = []
    changed = 0

    def _has_matching_caption_nearby(idx: int, fig_no: Optional[int]) -> bool:
        def _scan(delta: int, max_steps: int, max_non_empty: int) -> bool:
            non_empty = 0
            for step in range(1, max_steps + 1):
                j = idx + delta * step
                if not (0 <= j < len(lines)):
                    break
                raw = lines[j] or ""
                s = raw.strip()
                if not s:
                    continue
                if heading_re.match(raw) or img_re.match(raw):
                    break
                non_empty += 1
                m = cap_re.match(_normalize_text(s))
                if m:
                    if fig_no is None:
                        return True
                    try:
                        return int(m.group(1)) == int(fig_no)
                    except Exception:
                        return True
                if non_empty >= max_non_empty:
                    break
            return False

        return _scan(delta=1, max_steps=10, max_non_empty=3) or _scan(delta=-1, max_steps=8, max_non_empty=2)

    for idx, ln in enumerate(lines):
        out.append(ln)
        m = img_re.match(ln or "")
        if not m:
            continue
        asset_name = str(m.group(1) or "").strip()
        meta = figure_meta_by_asset.get(asset_name) if isinstance(figure_meta_by_asset, dict) else None
        if not isinstance(meta, dict):
            continue
        caption = _canonical_caption(meta)
        if not caption:
            continue
        fig_no = None
        try:
            fig_no = int(meta.get("fig_no"))
        except Exception:
            fig_no = None
        if _has_matching_caption_nearby(idx, fig_no):
            continue
        if out and out[-1].strip():
            out.append("")
        out.append(caption)
        out.append("")
        changed += 1

    if changed:
        try:
            print(f"[IMAGE_CAPTION_FIX] page {page_no}: injected {changed} caption(s) from metadata", flush=True)
        except Exception:
            pass
    return "\n".join(out)


def normalize_page_image_caption_order(
    md: str,
    *,
    page_index: int,
    figure_meta_by_asset: Optional[dict[str, dict]] = None,
) -> str:
    if not md or not figure_meta_by_asset:
        return md

    page_no = int(page_index) + 1
    img_re = re.compile(
        rf"^\s*!\[[^\]]*\]\(\./assets/(page_{page_no}_fig_\d+\.[^)]+)\)\s*$",
        flags=re.IGNORECASE,
    )
    cap_re = re.compile(
        r"^\s*(?:[*_`>#\[\(]\s*)*(?:Figure|Fig\.?)\s*(\d{1,4})(?:[A-Za-z])?\b",
        flags=re.IGNORECASE,
    )
    heading_re = re.compile(r"^\s*#{1,6}\s+")
    table_re = re.compile(r"^\s*\|")

    def _canonical_caption(meta: dict) -> Optional[str]:
        raw = _normalize_text(str(meta.get("caption") or "")).strip()
        if not raw:
            return None
        m = re.match(r"^\s*fig(?:ure)?\.?\s*(\d{1,4}[A-Za-z]?)\s*[.:]?\s*(.*)$", raw, flags=re.IGNORECASE)
        if not m:
            return raw
        ident = (m.group(1) or "").strip()
        tail = (m.group(2) or "").strip()
        if tail:
            return f"Figure {ident}. {tail}"
        return f"Figure {ident}."

    lines = md.splitlines()
    out: list[str] = []
    i = 0
    changed = 0
    while i < len(lines):
        ln = lines[i]
        m_img = img_re.match(ln or "")
        if not m_img:
            out.append(ln)
            i += 1
            continue

        asset_name = str(m_img.group(1) or "").strip()
        meta = figure_meta_by_asset.get(asset_name) if isinstance(figure_meta_by_asset, dict) else None
        out.append(ln)
        if not isinstance(meta, dict):
            i += 1
            continue

        caption = _canonical_caption(meta)
        fig_no = None
        try:
            fig_no = int(meta.get("fig_no"))
        except Exception:
            fig_no = None
        if not caption or fig_no is None:
            i += 1
            continue

        j = i + 1
        blanks_after = 0
        caption_below_idx = None
        while j < len(lines) and blanks_after <= 2:
            s = (lines[j] or "").strip()
            if not s:
                blanks_after += 1
                j += 1
                continue
            if heading_re.match(lines[j] or "") or img_re.match(lines[j] or ""):
                break
            m_cap = cap_re.match(_normalize_text(s))
            if m_cap:
                try:
                    if int(m_cap.group(1)) == fig_no:
                        caption_below_idx = j
                except Exception:
                    caption_below_idx = j
                break
            break

        if caption_below_idx is not None:
            i += 1
            continue

        prev_caption_idx = None
        prev_non_empty = 0
        k = len(out) - 2
        while k >= 0:
            raw = out[k] or ""
            s = raw.strip()
            if not s:
                k -= 1
                continue
            if img_re.match(raw) or heading_re.match(raw) or table_re.match(raw):
                break
            m_cap_prev = cap_re.match(_normalize_text(s))
            if m_cap_prev:
                prev_fig = None
                try:
                    prev_fig = int(m_cap_prev.group(1))
                except Exception:
                    prev_fig = None
                if prev_fig == fig_no:
                    prev_caption_idx = k
                break
            prev_non_empty += 1
            if prev_non_empty > 6:
                break
            k -= 1

        if prev_caption_idx is not None:
            image_line = out.pop()
            out = out[:prev_caption_idx] + [image_line, ""] + out[prev_caption_idx:]
            changed += 1
            i += 1
            continue

        if out and out[-1].strip():
            out.append("")
        out.append(caption)
        out.append("")
        changed += 1
        i += 1

    if changed:
        try:
            print(f"[IMAGE_CAPTION_ORDER] page {page_no}: normalized {changed} image/caption pair(s)", flush=True)
        except Exception:
            pass
    return "\n".join(out)


def repair_broken_image_links(md: str, *, save_dir: Path, assets_dir: Path) -> str:
    """
    Repair broken Markdown image links by remapping them to existing files in ./assets.
    This is a best-effort pass for vision-direct outputs where models may emit
    synthetic names like `figure_5.png` while extracted files are `page_*_fig_*.png`.
    """
    if not md:
        return md

    img_re = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
    matches = list(img_re.finditer(md))
    if not matches:
        return md

    cand_files = []
    try:
        for p in sorted(assets_dir.glob("*")):
            if (not p.is_file()):
                continue
            if p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
                continue
            cand_files.append(p)
    except Exception:
        cand_files = []
    if not cand_files:
        return md

    page_fig_re = re.compile(r"page_(\d+)_fig_(\d+)", flags=re.IGNORECASE)

    def _asset_key(p: Path) -> tuple[int, int, str]:
        m = page_fig_re.search(p.stem)
        if not m:
            return (10**9, 10**9, p.name.lower())
        try:
            return (int(m.group(1)), int(m.group(2)), p.name.lower())
        except Exception:
            return (10**9, 10**9, p.name.lower())

    cand_files = sorted(cand_files, key=_asset_key)

    refs: list[dict] = []
    used_names: set[str] = set()
    for m in matches:
        alt = m.group(1) or ""
        raw_path = (m.group(2) or "").strip().strip('"').strip("'")
        resolved_name = None
        if not (raw_path.startswith("http://") or raw_path.startswith("https://")):
            p_local = (save_dir / raw_path).resolve()
            if p_local.exists():
                resolved_name = p_local.name
            else:
                b = Path(raw_path).name
                if (assets_dir / b).exists():
                    resolved_name = b
        if resolved_name:
            used_names.add(resolved_name)
        refs.append(
            {
                "start": m.start(),
                "end": m.end(),
                "alt": alt,
                "raw_path": raw_path,
                "resolved_name": resolved_name,
            }
        )

    for r in refs:
        if r["resolved_name"]:
            continue
        b = Path(str(r["raw_path"])).name
        fig_no = None
        m_fig = re.search(r"figure[_\-\s]*(\d+)", b, flags=re.IGNORECASE)
        if m_fig:
            try:
                fig_no = int(m_fig.group(1))
            except Exception:
                fig_no = None
        chosen = None
        if fig_no is not None:
            exact = [
                p
                for p in cand_files
                if re.search(rf"(?:^|_)fig_{fig_no}(?:_|$|\.)", p.name, flags=re.IGNORECASE)
            ]
            if exact:
                for p in exact:
                    if p.name not in used_names:
                        chosen = p
                        break
                if chosen is None:
                    chosen = exact[0]
        if chosen is None:
            low = b.lower()
            sim = [p for p in cand_files if (low and low in p.name.lower())]
            if sim:
                for p in sim:
                    if p.name not in used_names:
                        chosen = p
                        break
                if chosen is None:
                    chosen = sim[0]
        if chosen is not None:
            r["resolved_name"] = chosen.name
            used_names.add(chosen.name)

    unresolved_idx = [i for i, r in enumerate(refs) if not r["resolved_name"]]
    if unresolved_idx:
        assigned_page_by_pos: dict[int, int] = {}
        for i, r in enumerate(refs):
            nm = r.get("resolved_name")
            if not nm:
                continue
            p = assets_dir / str(nm)
            m_pg = page_fig_re.search(p.stem)
            if m_pg:
                try:
                    assigned_page_by_pos[i] = int(m_pg.group(1))
                except Exception:
                    pass

        avail = [p for p in cand_files if p.name not in used_names]
        for i in unresolved_idx:
            if not avail:
                break
            prev_pages = [assigned_page_by_pos[j] for j in assigned_page_by_pos if j < i]
            next_pages = [assigned_page_by_pos[j] for j in assigned_page_by_pos if j > i]
            prev_p = max(prev_pages) if prev_pages else None
            next_p = min(next_pages) if next_pages else None

            def _page_of(p: Path) -> int:
                m = page_fig_re.search(p.stem)
                if not m:
                    return 10**9
                try:
                    return int(m.group(1))
                except Exception:
                    return 10**9

            choice = None
            if prev_p is not None and next_p is not None:
                in_range = [p for p in avail if prev_p <= _page_of(p) <= next_p]
                if in_range:
                    mid = (prev_p + next_p) / 2.0
                    choice = min(in_range, key=lambda p: abs(_page_of(p) - mid))
            if choice is None and prev_p is not None:
                after = [p for p in avail if _page_of(p) >= prev_p]
                if after:
                    choice = min(after, key=_page_of)
            if choice is None and next_p is not None:
                before = [p for p in avail if _page_of(p) <= next_p]
                if before:
                    choice = max(before, key=_page_of)
            if choice is None:
                choice = avail[0]

            refs[i]["resolved_name"] = choice.name
            used_names.add(choice.name)
            try:
                avail.remove(choice)
            except Exception:
                pass

    out = md
    shift = 0
    repaired = 0
    for r in refs:
        nm = r.get("resolved_name")
        if not nm:
            continue
        new_ref = f"![{r['alt']}](./assets/{nm})"
        s0 = int(r["start"]) + shift
        e0 = int(r["end"]) + shift
        old_ref = out[s0:e0]
        if old_ref != new_ref:
            out = out[:s0] + new_ref + out[e0:]
            shift += len(new_ref) - (e0 - s0)
            repaired += 1

    if repaired > 0:
        print(f"[IMAGE_FIX] repaired {repaired} image link(s)", flush=True)
    return out


def cleanup_page_local_image_markdown(md: str, *, page_index: int) -> str:
    if not md:
        return md

    lines = md.splitlines()

    def _line_norm(s: str) -> str:
        s = _normalize_text(s or "")
        s = re.sub(r"[*_`>#\[\]()]", "", s)
        s = re.sub(r"\s+", " ", s).strip().lower()
        return s

    img_re = re.compile(r"!\[([^\]]*)\]\((\.?/)?assets/[^)]+\)", flags=re.IGNORECASE)
    for idx, raw in enumerate(lines):
        m = img_re.search(raw or "")
        if not m:
            continue
        alt = (m.group(1) or "").strip()
        if not alt:
            continue
        fig_no = None
        for probe in range(idx, min(len(lines), idx + 4)):
            cap_line = _normalize_text(lines[probe] or "").strip()
            m_cap = re.match(r"^\*{0,2}(?:Figure|Fig\.?)\s*(\d+)\b", cap_line, flags=re.IGNORECASE)
            if m_cap:
                fig_no = m_cap.group(1)
                break
        if fig_no is None:
            m_alt = re.search(r"\b(?:Figure|Fig\.?)\s*(\d+)\b", alt, flags=re.IGNORECASE)
            if m_alt:
                fig_no = m_alt.group(1)
        if fig_no and (len(alt) > 32 or re.search(r"[.:,;]", alt)):
            new_ref = re.sub(
                r"!\[[^\]]*\]",
                f"![Figure {fig_no}]",
                raw,
                count=1,
            )
            lines[idx] = new_ref

    cleaned: list[str] = []
    recent_caption_keys: list[str] = []
    for raw in lines:
        st = _normalize_text(raw or "").strip()
        if not st:
            cleaned.append(raw)
            continue

        if page_index == 0:
            wide_gap = bool(re.search(r"(?:\s|\u2000|\u2001|\u2002|\u2003|\u2004|\u2005|\u2006|\u2007|\u2008|\u2009|\u200A|\u3000){6,}", raw or ""))
            bold_chunks = len(re.findall(r"\*\*[^*]{2,80}\*\*", raw or ""))
            wordish = len(re.findall(r"[A-Za-z]{2,}", st))
            if wide_gap and bold_chunks >= 2 and wordish <= 8 and "." not in st:
                continue

        if re.match(r"^\*{0,2}(?:Figure|Fig\.?)\s*\d+\b", st, flags=re.IGNORECASE):
            key = _line_norm(st)
            if key and key in recent_caption_keys[-3:]:
                continue
            recent_caption_keys.append(key)
        cleaned.append(raw)

    out = "\n".join(cleaned)
    out = re.sub(r"(\*\*(?:Figure|Fig\.?)\s*\d+\.?\*\*)\s+\*\*\s*", r"\1 ", out, flags=re.IGNORECASE)
    out = re.sub(r"(\*\*Table\s*\d+\.?\*\*)\s+\*\*\s*", r"\1 ", out, flags=re.IGNORECASE)
    return out


def normalize_page_local_image_link_order(
    md: str,
    *,
    page_index: int,
    image_names: list[str],
) -> str:
    """
    Normalize current-page figure link ordering to match extracted asset order.

    Vision models may reference valid assets but swap `fig_1` / `fig_2`.
    When the page contains the exact same set of current-page assets, remap
    links in document order to the sorted extracted order.
    """
    if not md:
        return md
    if len(image_names or []) < 2:
        return md

    page_no = int(page_index) + 1

    def _img_key(nm: str) -> tuple[int, str]:
        m = re.search(r"_fig_(\d+)", str(nm), flags=re.IGNORECASE)
        if m:
            try:
                return (int(m.group(1)), str(nm).lower())
            except Exception:
                pass
        return (10**9, str(nm).lower())

    ordered_assets = sorted({str(n).strip() for n in image_names if str(n).strip()}, key=_img_key)
    if len(ordered_assets) < 2:
        return md

    img_re = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
    page_fig_re = re.compile(rf"^page_{page_no}_fig_(\d+)\.[A-Za-z0-9]+$", flags=re.IGNORECASE)
    refs: list[dict] = []
    for m in img_re.finditer(md):
        raw_path = (m.group(2) or "").strip().strip('"').strip("'")
        base = Path(raw_path).name
        if not page_fig_re.match(base):
            continue
        refs.append({"start": m.start(), "end": m.end(), "alt": m.group(1) or "", "path": raw_path, "base": base})

    if len(refs) < 2:
        return md
    if len(refs) != len(ordered_assets):
        return md

    current_order = [str(r["base"]) for r in refs]
    if set(current_order) != set(ordered_assets):
        return md
    if current_order == ordered_assets:
        return md

    out = md
    shift = 0
    changed = 0
    for r, target_name in zip(refs, ordered_assets):
        new_ref = f"![{r['alt']}](./assets/{target_name})"
        s0 = int(r["start"]) + shift
        e0 = int(r["end"]) + shift
        old_ref = out[s0:e0]
        if old_ref == new_ref:
            continue
        out = out[:s0] + new_ref + out[e0:]
        shift += len(new_ref) - (e0 - s0)
        changed += 1
    if changed:
        try:
            print(f"[IMAGE_ORDER] page {page_no}: normalized {changed} figure link(s)", flush=True)
        except Exception:
            pass
    return out


def extract_figure_number_from_text(text: str) -> Optional[int]:
    t = _normalize_text(text or "").strip()
    if not t:
        return None
    m = re.search(
        r"\b(?:fig(?:ure)?\.?)\s*(\d{1,4})(?:[A-Za-z])?\b",
        t,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def figure_remap_debug_enabled() -> bool:
    try:
        return bool(int(os.environ.get("KB_PDF_DEBUG_FIG_REMAP", "0") or "0"))
    except Exception:
        return False


def reorder_page_figure_pairs_by_number(
    md: str,
    *,
    page_index: int,
) -> str:
    """
    Reorder adjacent figure image+caption pairs for the same page by figure number.
    This keeps visual order stable when VL outputs Fig.3 before Fig.2 on one page.
    """
    if not md:
        return md

    page_no = int(page_index) + 1
    img_re = re.compile(
        rf"^\s*!\[[^\]]*\]\(\./assets/(page_{page_no}_fig_\d+\.[^)]+)\)\s*$",
        flags=re.IGNORECASE,
    )
    cap_re = re.compile(
        r"^\s*(?:[*_`>#\[\(]\s*)*(?:fig(?:ure)?\.?)\s*(\d{1,4})(?:[A-Za-z])?\b",
        flags=re.IGNORECASE,
    )
    heading_re = re.compile(r"^\s*#{1,6}\s+")

    lines = md.splitlines()
    n = len(lines)
    blocks: list[dict] = []
    i = 0
    while i < n:
        m_img = img_re.match(lines[i] or "")
        if not m_img:
            i += 1
            continue

        caption_idx = None
        fig_no = None
        j = i + 1
        scanned = 0
        while j < n and scanned < 8:
            s = (lines[j] or "").strip()
            if not s:
                j += 1
                scanned += 1
                continue
            if img_re.match(lines[j] or "") or heading_re.match(lines[j] or ""):
                break
            m_cap = cap_re.match(_normalize_text(s))
            if m_cap:
                caption_idx = j
                try:
                    fig_no = int(m_cap.group(1))
                except Exception:
                    fig_no = None
                break
            break

        if caption_idx is None or fig_no is None:
            i += 1
            continue

        end = caption_idx + 1
        cont = 0
        while end < n:
            s = (lines[end] or "").strip()
            if not s:
                end += 1
                break
            if img_re.match(lines[end] or "") or heading_re.match(lines[end] or ""):
                break
            if cont >= 2:
                break
            end += 1
            cont += 1

        blocks.append(
            {
                "start": int(i),
                "end": int(end),
                "fig_no": int(fig_no),
                "img_name": str(m_img.group(1)),
            }
        )
        i = end

    if len(blocks) < 2:
        return md

    runs: list[list[dict]] = []
    cur: list[dict] = [blocks[0]]
    for b in blocks[1:]:
        prev = cur[-1]
        gap_lines = lines[int(prev["end"]):int(b["start"])]
        only_blank_gap = all((not (gl or "").strip()) for gl in gap_lines)
        if only_blank_gap:
            cur.append(b)
        else:
            if len(cur) >= 2:
                runs.append(cur)
            cur = [b]
    if len(cur) >= 2:
        runs.append(cur)
    if not runs:
        return md

    changed = 0
    out_lines: list[str] = []
    cursor = 0
    for run in runs:
        seg_start = int(run[0]["start"])
        seg_end = int(run[-1]["end"])
        out_lines.extend(lines[cursor:seg_start])
        nums = [int(x["fig_no"]) for x in run]
        if nums == sorted(nums):
            out_lines.extend(lines[seg_start:seg_end])
            cursor = seg_end
            continue

        run_sorted = sorted(run, key=lambda x: (int(x["fig_no"]), int(x["start"])))
        chunk: list[str] = []
        for idx, rb in enumerate(run_sorted):
            chunk.extend(lines[int(rb["start"]):int(rb["end"])])
            if idx != len(run_sorted) - 1 and (not chunk or chunk[-1].strip()):
                chunk.append("")
        out_lines.extend(chunk)
        cursor = seg_end
        changed += 1

    out_lines.extend(lines[cursor:])
    if changed:
        try:
            print(f"[FIG_ORDER] page {page_no}: reordered {changed} figure block run(s) by figure number", flush=True)
        except Exception:
            pass
    return "\n".join(out_lines)


def remap_page_image_links_by_caption(
    md: str,
    *,
    page_index: int,
    figure_meta_by_asset: Optional[dict[str, dict]] = None,
) -> str:
    """
    Use per-page figure metadata (asset -> figure number) to remap image links
    according to nearby caption numbers in markdown.
    """
    if not md:
        return md
    if not figure_meta_by_asset:
        return md

    fig_to_asset: dict[int, str] = {}
    ambiguous: set[int] = set()
    for asset_name, meta in (figure_meta_by_asset or {}).items():
        if not isinstance(meta, dict):
            continue
        try:
            fig_no = int(meta.get("fig_no"))
        except Exception:
            continue
        if fig_no in fig_to_asset and fig_to_asset.get(fig_no) != asset_name:
            ambiguous.add(fig_no)
            continue
        fig_to_asset[fig_no] = str(asset_name)
    for n in ambiguous:
        fig_to_asset.pop(n, None)
    if not fig_to_asset:
        return md

    page_no = int(page_index) + 1
    page_fig_re = re.compile(rf"^page_{page_no}_fig_(\d+)\.[A-Za-z0-9]+$", flags=re.IGNORECASE)
    img_re = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
    heading_re = re.compile(r"^\s*#{1,6}\s+")
    cap_re = re.compile(
        r"^\s*(?:[*_`>#\[\(]\s*)*(?:fig(?:ure)?\.?)\s*(\d{1,4})(?:[A-Za-z])?\b",
        flags=re.IGNORECASE,
    )

    lines = md.splitlines()
    changed = 0
    debug = figure_remap_debug_enabled()
    trace: list[str] = []
    if debug:
        try:
            pairs = ", ".join(f"{k}->{v}" for k, v in sorted(fig_to_asset.items(), key=lambda x: x[0]))
            trace.append(f"[FIG_DEBUG] page {page_no} mapping: {pairs or '<empty>'}")
        except Exception:
            pass

    def _find_nearby_caption_fig(line_idx: int) -> Optional[int]:
        def _scan(*, delta: int, max_steps: int, max_non_empty: int) -> Optional[int]:
            non_empty = 0
            for step in range(1, max_steps + 1):
                j = line_idx + delta * step
                if not (0 <= j < len(lines)):
                    break
                raw = lines[j] or ""
                s = raw.strip()
                if not s:
                    continue
                if heading_re.match(raw) or img_re.search(raw):
                    break
                non_empty += 1
                m = cap_re.match(_normalize_text(s))
                if m:
                    try:
                        return int(m.group(1))
                    except Exception:
                        return None
                if non_empty >= max_non_empty:
                    break
            return None

        down = _scan(delta=1, max_steps=8, max_non_empty=3)
        if down is not None:
            return down
        up = _scan(delta=-1, max_steps=6, max_non_empty=2)
        if up is not None:
            return up
        return None

    out_lines: list[str] = []
    for idx, line in enumerate(lines):
        matches = list(img_re.finditer(line))
        if not matches:
            out_lines.append(line)
            continue
        out = line
        shift = 0
        for m in matches:
            alt = m.group(1) or ""
            raw_path = (m.group(2) or "").strip().strip('"').strip("'")
            base = Path(raw_path).name
            if not page_fig_re.match(base):
                continue

            fig_no = extract_figure_number_from_text(alt)
            if fig_no is None:
                fig_no = _find_nearby_caption_fig(idx)
            if fig_no is None:
                if debug:
                    trace.append(
                        f"[FIG_DEBUG] page {page_no} line {idx+1}: skip {base} (no fig number from alt/caption)"
                    )
                continue

            target = fig_to_asset.get(int(fig_no))
            if not target or target == base:
                if debug:
                    target_show = target if target else "<none>"
                    trace.append(
                        f"[FIG_DEBUG] page {page_no} line {idx+1}: keep {base} (fig_no={fig_no}, target={target_show})"
                    )
                continue

            new_ref = f"![{alt}](./assets/{target})"
            s0 = int(m.start()) + shift
            e0 = int(m.end()) + shift
            old_ref = out[s0:e0]
            if old_ref == new_ref:
                continue
            out = out[:s0] + new_ref + out[e0:]
            shift += len(new_ref) - (e0 - s0)
            changed += 1
            if debug:
                trace.append(
                    f"[FIG_DEBUG] page {page_no} line {idx+1}: remap {base} -> {target} (fig_no={fig_no})"
                )
        out_lines.append(out)

    if debug and trace:
        for row in trace:
            try:
                print(row, flush=True)
            except Exception:
                pass
    if changed:
        try:
            print(f"[IMAGE_REMAP] page {page_no}: remapped {changed} figure link(s) by caption metadata", flush=True)
        except Exception:
            pass
    return "\n".join(out_lines)
