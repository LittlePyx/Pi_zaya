from __future__ import annotations

import os
from pathlib import Path


_NON_PAPER_SOURCE_STEMS = {
    "quality_report",
    "assets_manifest",
    "output",
}

_NON_PAPER_SOURCE_NAMES = {
    "quality_report.md",
    "quality_report.en.md",
    "quality_report.pdf",
    "assets_manifest.md",
    "assets_manifest.pdf",
    "output.md",
    "output.pdf",
}

_NON_PAPER_SOURCE_DIRS = {
    ".conversion_cache",
}


def is_nonpaper_artifact_source(source_path: str) -> bool:
    raw = str(source_path or "").strip()
    if not raw:
        return True
    p = Path(raw)
    low_name = p.name.strip().lower()
    low_stem = p.stem.strip().lower()
    low_parts = {str(part or "").strip().lower() for part in p.parts}
    if low_parts & _NON_PAPER_SOURCE_DIRS:
        return True
    if low_name in _NON_PAPER_SOURCE_NAMES:
        return True
    if low_stem.endswith(".en"):
        low_stem = low_stem[:-3]
    if low_stem in _NON_PAPER_SOURCE_STEMS:
        return True
    if "markdown quality analysis report" in low_name:
        return True
    if low_name.startswith("page_") and low_name.endswith("_original_tables.md"):
        return True
    return False


def is_excluded_source_path(source_path: str) -> bool:
    return is_nonpaper_artifact_source(source_path)


def is_temporary_source_path(source_path: str) -> bool:
    """Return whether a retrieval source is a transient conversion artifact."""

    raw = str(source_path or "").strip()
    if not raw:
        return True
    # An explicitly configured DB root is authoritative even when one of its
    # parent directories happens to use a temporary-looking name.
    try:
        raw_db_dir = str(os.environ.get("KB_DB_DIR", "") or "").strip()
        if raw_db_dir:
            db_dir = Path(raw_db_dir).expanduser().resolve()
            source = Path(raw).expanduser().resolve()
            if source.is_relative_to(db_dir):
                return False
    except Exception:
        pass
    if is_excluded_source_path(raw):
        return True
    path = Path(raw)
    low_parts = [str(part).strip().lower() for part in path.parts]
    low_name = path.name.lower()
    low_stem = path.stem.lower()
    if any(part in {"temp", "__pycache__"} for part in low_parts):
        return True
    if any(
        part.startswith("__upload__")
        or part.startswith("_tmp_")
        or part.startswith("tmp_")
        for part in low_parts
    ):
        return True
    if low_name.startswith("__upload__") or low_stem.startswith("__upload__"):
        return True
    if low_name.startswith("_tmp_") or low_stem.startswith("_tmp_"):
        return True
    if low_name.startswith("tmp_") or low_stem.startswith("tmp_"):
        return True
    return False
