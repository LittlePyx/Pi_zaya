from __future__ import annotations

from pathlib import Path
from typing import Callable

from api.reference_source_identity import (
    _same_source_identity,
    _same_source_title_identity,
    _source_filename,
)
from kb.file_naming import citation_meta_display_pdf_name


def _hit_matches_guide_source(meta: dict, *, guide_source_path: str, guide_source_name: str) -> bool:
    if not isinstance(meta, dict):
        return False
    candidates = [
        str(meta.get("source_path") or "").strip(),
        str(meta.get("source_name") or "").strip(),
        str(meta.get("display_name") or "").strip(),
    ]
    candidates = [item for item in candidates if item]
    if not candidates:
        return False
    guide_path = str(guide_source_path or "").strip()
    guide_name = str(guide_source_name or "").strip()
    for candidate in candidates:
        if guide_path and _same_source_identity(candidate, guide_path):
            return True
        if guide_name and _same_source_title_identity(candidate, guide_name):
            return True
        if guide_path and _same_source_title_identity(candidate, guide_path):
            return True
    return False


def _display_source_name(
    source_path: str,
    pdf_path: Path | None,
    lib_store: object | None,
    *,
    debug_log: Callable[[str], None] | None = None,
) -> str:
    try:
        if pdf_path is not None and lib_store is not None:
            meta = lib_store.get_citation_meta(pdf_path)  # type: ignore[attr-defined]
            full_name = citation_meta_display_pdf_name(meta)
            if full_name:
                return full_name
    except Exception as exc:
        if callable(debug_log):
            debug_log(f"[refs] display_name fallback for {str(source_path or '')[-80:]}: {exc}")

    name = _source_filename(source_path) or str(source_path or "")
    low = name.lower()
    if low.endswith(".en.md"):
        name = name[:-6] + ".pdf"
    elif low.endswith(".md"):
        name = name[:-3] + ".pdf"
    return name or "unknown.pdf"


__all__ = [
    "_display_source_name",
    "_hit_matches_guide_source",
]
