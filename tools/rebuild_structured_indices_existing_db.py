from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _force_utf8_stdout() -> None:
    """
    Windows PowerShell often defaults to a legacy code page (e.g. gbk) which can
    crash when printing paths that contain non-encodable characters.
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[attr-defined]
    except Exception:
        # Best effort. If reconfigure is unavailable, we at least avoid crashing
        # by letting Python handle output as-is.
        pass

try:
    from kb.config import load_settings
    from kb.converter.structured_indices import rebuild_structured_indices_for_markdown
except ModuleNotFoundError:  # pragma: no cover
    # Support running the script directly without installing the project as a package.
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))
    from kb.config import load_settings  # type: ignore
    from kb.converter.structured_indices import rebuild_structured_indices_for_markdown  # type: ignore


@dataclass
class IndexCheck:
    md_path: Path
    assets_dir: Path
    status: str
    reason: str
    md_mtime: float
    index_mtime_min: float
    index_mtime_max: float


INDEX_FILES = [
    "anchor_index.json",
    "equation_index.json",
    "reference_index.json",
    "figure_index.json",
]


def _mtime(p: Path) -> float:
    try:
        return float(p.stat().st_mtime)
    except Exception:
        return 0.0


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _check_one(md_path: Path) -> IndexCheck:
    assets_dir = md_path.parent / "assets"
    md_mtime = _mtime(md_path)
    mtimes: list[float] = []
    missing: list[str] = []
    bad_doc: list[str] = []
    for name in INDEX_FILES:
        p = assets_dir / name
        if not p.exists():
            missing.append(name)
            continue
        mtimes.append(_mtime(p))
        payload = _load_json(p)
        doc_path = str(payload.get("doc_path") or "").strip()
        if doc_path:
            # The index stores resolved path; compare case-insensitively on Windows.
            if Path(doc_path).resolve(strict=False) != md_path.resolve(strict=False):
                bad_doc.append(name)
    if missing:
        return IndexCheck(
            md_path=md_path,
            assets_dir=assets_dir,
            status="stale",
            reason=f"missing={','.join(missing)}",
            md_mtime=md_mtime,
            index_mtime_min=min(mtimes) if mtimes else 0.0,
            index_mtime_max=max(mtimes) if mtimes else 0.0,
        )
    if bad_doc:
        return IndexCheck(
            md_path=md_path,
            assets_dir=assets_dir,
            status="stale",
            reason=f"doc_path_mismatch={','.join(bad_doc)}",
            md_mtime=md_mtime,
            index_mtime_min=min(mtimes) if mtimes else 0.0,
            index_mtime_max=max(mtimes) if mtimes else 0.0,
        )
    if mtimes and max(mtimes) + 1e-6 < md_mtime:
        return IndexCheck(
            md_path=md_path,
            assets_dir=assets_dir,
            status="stale",
            reason="index_older_than_md",
            md_mtime=md_mtime,
            index_mtime_min=min(mtimes),
            index_mtime_max=max(mtimes),
        )
    return IndexCheck(
        md_path=md_path,
        assets_dir=assets_dir,
        status="ok",
        reason="",
        md_mtime=md_mtime,
        index_mtime_min=min(mtimes) if mtimes else 0.0,
        index_mtime_max=max(mtimes) if mtimes else 0.0,
    )


def _iter_md_files(db_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for p in db_dir.rglob("*.md"):
        name = p.name.lower()
        # Skip analysis artifacts and non-guide markdown.
        if name in {"quality_report.md", "report.md"}:
            continue
        if name.endswith(".en.md"):
            paths.append(p)
    paths.sort(key=lambda x: str(x))
    return paths


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdout()
    ap = argparse.ArgumentParser(description="Audit/rebuild structured index json files for existing markdown guides.")
    ap.add_argument("--db-dir", default="", help="Override KB_DB_DIR; defaults to settings.db_dir")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of md files scanned (0=all)")
    ap.add_argument("--stale-only", action="store_true", help="Only print stale entries")
    ap.add_argument("--apply", action="store_true", help="Rebuild indices for stale entries")
    ap.add_argument("--max-rebuild", type=int, default=0, help="Max rebuild count (0=unlimited)")
    args = ap.parse_args(argv)

    settings = load_settings()
    db_dir = Path(str(args.db_dir or settings.db_dir)).expanduser().resolve()
    if not db_dir.exists():
        print(f"[ERR] db_dir not found: {db_dir}", file=sys.stderr)
        return 2

    md_files = _iter_md_files(db_dir)
    if int(args.limit or 0) > 0:
        md_files = md_files[: int(args.limit)]

    stale: list[IndexCheck] = []
    ok = 0
    for md in md_files:
        chk = _check_one(md)
        if chk.status == "ok":
            ok += 1
            if not args.stale_only:
                print(f"ok  {md}")
            continue
        stale.append(chk)
        print(f"stale  {md}  reason={chk.reason}")

    print(f"summary ok={ok} stale={len(stale)} total={len(md_files)}")

    if not args.apply:
        return 0

    rebuilt = 0
    for chk in stale:
        if int(args.max_rebuild or 0) > 0 and rebuilt >= int(args.max_rebuild):
            break
        try:
            rebuild_structured_indices_for_markdown(chk.md_path, assets_dir=chk.assets_dir)
            rebuilt += 1
            print(f"rebuilt {chk.md_path}")
        except Exception as e:
            print(f"[WARN] rebuild failed {chk.md_path}: {e}", file=sys.stderr)
    print(f"rebuilt_count={rebuilt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
