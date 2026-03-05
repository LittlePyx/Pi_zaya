from __future__ import annotations

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


def is_nonpaper_artifact_source(source_path: str) -> bool:
    raw = str(source_path or "").strip()
    if not raw:
        return True
    p = Path(raw)
    low_name = p.name.strip().lower()
    low_stem = p.stem.strip().lower()
    if low_name in _NON_PAPER_SOURCE_NAMES:
        return True
    if low_stem.endswith(".en"):
        low_stem = low_stem[:-3]
    if low_stem in _NON_PAPER_SOURCE_STEMS:
        return True
    if "markdown quality analysis report" in low_name:
        return True
    return False


def is_excluded_source_path(source_path: str) -> bool:
    return is_nonpaper_artifact_source(source_path)
