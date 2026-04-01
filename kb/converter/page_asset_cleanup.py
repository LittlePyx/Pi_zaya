from __future__ import annotations

import json
import re
from pathlib import Path


def cleanup_stale_page_assets(*, cfg, assets_dir: Path, total_pages: int) -> None:
    """
    Remove stale page-scoped assets in the target conversion range so reruns
    don't keep old split/cropped figure files that are no longer referenced.
    """
    start = max(0, int(getattr(cfg, "start_page", 0) or 0))
    end = int(getattr(cfg, "end_page", -1) or -1)
    if end < 0:
        end = int(total_pages)
    end = min(int(total_pages), int(end))
    if start >= end:
        return

    page_set = {int(i) for i in range(start + 1, end + 1)}  # filenames are 1-based
    pat = re.compile(
        r"^page_(\d+)_(?:fig|eq)_\d+\.(?:png|jpg|jpeg|webp|gif)$",
        re.IGNORECASE,
    )
    pat_meta = re.compile(
        r"^page_(\d+)_fig_\d+\.meta\.json$",
        re.IGNORECASE,
    )
    pat_page_index = re.compile(
        r"^page_(\d+)_fig_index\.json$",
        re.IGNORECASE,
    )
    removed = 0
    for p in assets_dir.glob("page_*"):
        try:
            m = pat.match(p.name) or pat_meta.match(p.name) or pat_page_index.match(p.name)
            if not m:
                continue
            page_no = int(m.group(1))
            if page_no not in page_set:
                continue
            p.unlink(missing_ok=True)
            removed += 1
        except Exception:
            continue
    if removed > 0:
        print(
            f"[ASSET_CLEAN] removed {removed} stale page asset(s) in range {start+1}-{end}",
            flush=True,
        )


def cleanup_unreferenced_assets(md: str, *, assets_dir: Path) -> None:
    if not md or not assets_dir.exists():
        return

    used: set[str] = set()
    for m in re.finditer(r"!\[[^\]]*\]\(\./assets/([^)]+)\)", md or "", flags=re.IGNORECASE):
        nm = str(m.group(1) or "").strip()
        if nm:
            used.add(nm)
    if not used:
        return

    page_index_files = list(assets_dir.glob("page_*_fig_index.json"))
    removed = 0
    for p in assets_dir.glob("page_*_fig_*.*"):
        if p.name.endswith("_index.json"):
            continue
        asset_name = p.name
        if p.suffix.lower() == ".json" and p.stem.endswith(".meta"):
            asset_name = p.stem[:-5] + ".png"
        if asset_name in used:
            continue
        try:
            p.unlink(missing_ok=True)
            removed += 1
        except Exception:
            continue

    for idx_path in page_index_files:
        try:
            payload = json.loads(idx_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        figures = payload.get("figures") or []
        kept = [fig for fig in figures if str(fig.get("asset_name") or "") in used]
        if kept:
            payload["figures"] = kept
            try:
                idx_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass
        else:
            try:
                idx_path.unlink(missing_ok=True)
            except Exception:
                pass
    if removed > 0:
        print(f"[ASSET_CLEAN] removed {removed} unreferenced asset(s)", flush=True)
