from __future__ import annotations

import hashlib
import os
import shutil
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from typing import Optional


def _to_os_path(path_like: Path | str) -> str:
    p = Path(path_like).expanduser()
    try:
        s = str(p.resolve(strict=False))
    except Exception:
        s = str(p)
    if os.name != "nt":
        return s
    if s.startswith("\\\\?\\"):
        return s
    if s.startswith("\\\\"):
        tail = s.lstrip("\\")
        return "\\\\?\\UNC\\" + tail
    return "\\\\?\\" + s


def _path_exists(path_like: Path | str) -> bool:
    try:
        return bool(os.path.exists(_to_os_path(path_like)))
    except Exception:
        try:
            return Path(path_like).exists()
        except Exception:
            return False


def _path_is_file(path_like: Path | str) -> bool:
    try:
        return bool(os.path.isfile(_to_os_path(path_like)))
    except Exception:
        try:
            return Path(path_like).is_file()
        except Exception:
            return False


def _path_is_dir(path_like: Path | str) -> bool:
    try:
        return bool(os.path.isdir(_to_os_path(path_like)))
    except Exception:
        try:
            return Path(path_like).is_dir()
        except Exception:
            return False


def _path_mtime(path_like: Path | str) -> float:
    try:
        return float(os.path.getmtime(_to_os_path(path_like)))
    except Exception:
        try:
            return float(Path(path_like).stat().st_mtime)
        except Exception:
            return 0.0


def _replace_or_copy(src: Path, dest: Path) -> bool:
    src_os = _to_os_path(src)
    dst_os = _to_os_path(dest)
    try:
        os.makedirs(os.path.dirname(dst_os), exist_ok=True)
    except Exception:
        pass
    try:
        if os.path.exists(dst_os):
            os.remove(dst_os)
    except Exception:
        pass
    try:
        os.replace(src_os, dst_os)
        return True
    except Exception:
        pass
    try:
        with open(src_os, "rb") as fr, open(dst_os, "wb") as fw:
            shutil.copyfileobj(fr, fw, length=1024 * 1024)
        return True
    except Exception:
        return False


def _is_ignored_md_dir_name(name: str) -> bool:
    n = (name or "").strip().lower()
    if n in {"chunks", "temp", "__pycache__", ".git"}:
        return True
    if n.startswith("__upload__"):
        return True
    if n.startswith("_tmp_") or n.startswith("tmp_"):
        return True
    return False


def _would_hit_windows_path_limit(path_obj: Path, *, safe_limit: int = 245) -> bool:
    """
    Conservative check for legacy Windows MAX_PATH environments.
    Even if long-path support is enabled, using a safe cap avoids flaky renames.
    """
    try:
        return (os.name == "nt") and (len(str(path_obj)) >= int(safe_limit))
    except Exception:
        return False

def _resolve_md_output_paths(out_root: Path, pdf_path: Path) -> tuple[Path, Path, bool]:
    pdf = Path(pdf_path)
    md_folder = Path(out_root) / pdf.stem
    canonical = md_folder / f"{pdf.stem}.en.md"
    out_md = md_folder / "output.md"

    def _mtime(p: Path) -> float:
        return _path_mtime(p)

    # Prefer the newest main markdown among canonical/output when both exist.
    if _path_exists(canonical) and _path_exists(out_md):
        md_main = out_md if _mtime(out_md) >= _mtime(canonical) else canonical
        return md_folder, md_main, True

    # Prefer output.md when present (converter may keep it when Windows rename fails).
    if _path_exists(out_md):
        return md_folder, out_md, True

    if _path_exists(canonical):
        return md_folder, canonical, True

    # Fallback: pick newest *.md in the folder.
    if _path_exists(md_folder):
        try:
            cands = [
                x
                for x in md_folder.glob("*.md")
                if _path_is_file(x) and x.name.lower() not in {"assets_manifest.md"}
            ]
            if cands:
                cands.sort(key=_mtime, reverse=True)
                return md_folder, cands[0], True
        except Exception:
            pass

    return md_folder, canonical, False

def _next_pdf_dest_path(pdf_dir: Path, base_name: str, *, max_suffix: int = 100) -> Path:
    dest_pdf = Path(pdf_dir) / f"{base_name}.pdf"
    if not dest_pdf.exists():
        return dest_pdf
    k = 2
    while (Path(pdf_dir) / f"{base_name}-{k}.pdf").exists() and k < int(max_suffix):
        k += 1
    return Path(pdf_dir) / f"{base_name}-{k}.pdf"

def _persist_upload_pdf(tmp_path: Path, dest_pdf: Path, data: bytes) -> None:
    try:
        if tmp_path.exists() and tmp_path.resolve() != dest_pdf.resolve():
            tmp_path.replace(dest_pdf)
            return
    except Exception:
        pass
    dest_pdf.write_bytes(data)

def _write_tmp_upload(pdf_dir: Path, filename: str, data: bytes) -> Path:
    stem = (Path(filename).stem or "upload").strip() or "upload"
    tmp = pdf_dir / f"__upload__{stem}.pdf"
    tmp.write_bytes(data)
    return tmp

def _sha1_bytes(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()

def _pick_directory_dialog(initial_dir: str) -> Optional[str]:
    """
    Open a native folder picker on the local machine.
    """
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        sel = filedialog.askdirectory(initialdir=initial_dir or None, title="选择目录")
        try:
            root.destroy()
        except Exception:
            pass
        sel = (sel or "").strip()
        return sel or None
    except Exception:
        return None

def _cleanup_tmp_uploads(pdf_dir: Path) -> int:
    n = 0
    try:
        for p in Path(pdf_dir).glob("__upload__*.pdf"):
            try:
                p.unlink()
                n += 1
            except Exception:
                pass
    except Exception:
        pass
    return n


def _cleanup_tmp_md_artifacts(md_out_root: Path) -> tuple[int, list[str]]:
    """
    Remove temporary markdown artifacts produced during upload/probing.
    """
    md_root = Path(md_out_root)
    if (not md_root.exists()) or (not md_root.is_dir()):
        return 0, []

    removed: list[str] = []
    try:
        for p in md_root.iterdir():
            name = p.name
            low = name.lower()
            is_tmp = low.startswith("__upload__") or low.startswith("_tmp_") or low.startswith("tmp_")
            if not is_tmp:
                continue
            try:
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink(missing_ok=True)  # type: ignore[call-arg]
                removed.append(name)
            except Exception:
                continue
    except Exception:
        return len(removed), removed
    return len(removed), removed


def _list_orphan_md_dirs(md_out_root: Path, pdf_dir: Path) -> list[Path]:
    """
    Find generated markdown folders that no longer have a matching PDF stem.
    We only consider directories that look like app-generated conversion output:
    <folder>/<folder>.en.md exists.
    """
    md_root = Path(md_out_root)
    pdf_root = Path(pdf_dir)
    if (not _path_exists(md_root)) or (not _path_is_dir(md_root)) or (not _path_exists(pdf_root)):
        return []

    try:
        pdf_stems = {p.stem.strip().lower() for p in pdf_root.glob("*.pdf") if p.is_file()}
    except Exception:
        pdf_stems = set()
    if not pdf_stems:
        return []

    out: list[Path] = []
    try:
        for d in md_root.iterdir():
            if (not _path_is_dir(d)) or _is_ignored_md_dir_name(d.name):
                continue
            if d.name.strip().lower() in pdf_stems:
                continue

            main_md = d / f"{d.name}.en.md"
            if not _path_exists(main_md):
                continue
            out.append(d)
    except Exception:
        return out
    return out


def _stash_orphan_md_dirs(md_out_root: Path, pdf_dir: Path) -> tuple[int, list[str]]:
    """
    Move orphan markdown folders to temp stash instead of deleting.
    The ingest script excludes `temp/` by default, so these files are ignored safely.
    """
    orphans = _list_orphan_md_dirs(md_out_root, pdf_dir)
    if not orphans:
        return 0, []

    md_root = Path(md_out_root)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    stash_root = md_root / "temp" / "_orphan_md_stash" / stamp
    stash_root.mkdir(parents=True, exist_ok=True)

    moved: list[str] = []
    for d in orphans:
        try:
            dest = stash_root / d.name
            if dest.exists():
                k = 2
                while (stash_root / f"{d.name}-{k}").exists() and k < 999:
                    k += 1
                dest = stash_root / f"{d.name}-{k}"
            shutil.move(str(d), str(dest))
            moved.append(d.name)
        except Exception:
            continue
    return len(moved), moved


def _find_md_main_name_mismatches(md_out_root: Path, pdf_dir: Path) -> list[tuple[Path, Path]]:
    """
    Detect folders whose canonical main markdown `<stem>.en.md` is missing,
    but another markdown file exists and can be renamed to canonical name.
    """
    md_root = Path(md_out_root)
    pdf_root = Path(pdf_dir)
    if (not _path_exists(md_root)) or (not _path_exists(pdf_root)):
        return []

    out: list[tuple[Path, Path]] = []
    try:
        for p in pdf_root.glob("*.pdf"):
            if not _path_is_file(p):
                continue
            folder = md_root / p.stem
            if (not _path_exists(folder)) or (not _path_is_dir(folder)):
                continue
            canonical = folder / f"{p.stem}.en.md"
            if _path_exists(canonical):
                continue

            candidates = [
                x
                for x in sorted(folder.glob("*.md"))
                if _path_is_file(x) and x.name.lower() != "assets_manifest.md" and x.name != canonical.name
            ]
            if not candidates:
                continue

            preferred = [x for x in candidates if x.name.lower().endswith(".en.md")]
            src = preferred[0] if preferred else candidates[0]
            out.append((src, canonical))
    except Exception:
        return out
    return out


def _sync_md_main_filenames(md_out_root: Path, pdf_dir: Path) -> tuple[int, list[str]]:
    """
    Rename markdown main files to canonical `<pdf_stem>.en.md` inside matched folders.
    """
    pairs = _find_md_main_name_mismatches(md_out_root, pdf_dir)
    if not pairs:
        return 0, []

    renamed: list[str] = []
    for src, dest in pairs:
        try:
            if _path_exists(dest):
                continue
            if _replace_or_copy(src, dest):
                renamed.append(f"{src.parent.name}: {src.name} -> {dest.name}")
        except Exception:
            continue
    return len(renamed), renamed

def _list_pdf_paths_fast(pdf_dir: Path) -> list[Path]:
    """
    Fast non-recursive PDF listing.
    Avoids per-file Path.stat() calls (important on large folders / slow disks).
    """
    pdf_dir = Path(pdf_dir)
    out: list[Path] = []
    try:
        with os.scandir(pdf_dir) as it:
            for e in it:
                try:
                    if not e.is_file():
                        continue
                    if not e.name.lower().endswith(".pdf"):
                        continue
                    out.append(Path(e.path))
                except Exception:
                    continue
    except Exception:
        # Fallback for unusual FS errors.
        try:
            out = [x for x in pdf_dir.glob("*.pdf") if x.is_file()]
        except Exception:
            out = []
    return out
