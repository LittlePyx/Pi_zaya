from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class LibraryPaths:
    pdf_dir: Path
    md_dir: Path
    pdf_source: str
    md_source: str

    def public_payload(self, *, reveal_paths: bool = True) -> dict[str, object]:
        return {
            "pdf_dir": str(self.pdf_dir) if reveal_paths else "",
            "md_dir": str(self.md_dir) if reveal_paths else "",
            "pdf_source": self.pdf_source,
            "md_source": self.md_source,
            "uses_managed_defaults": self.pdf_source != "preference" or self.md_source != "preference",
        }


def _resolved_path(
    *,
    preference: object,
    environment: object,
    fallback: Path,
) -> tuple[Path, str]:
    preferred = str(preference or "").strip()
    if preferred:
        return Path(preferred).expanduser().resolve(), "preference"
    configured = str(environment or "").strip()
    if configured:
        return Path(configured).expanduser().resolve(), "environment"
    return fallback.expanduser().resolve(), "default"


def resolve_library_paths(
    settings,
    prefs: Mapping[str, object] | None = None,
    environ: Mapping[str, str] | None = None,
) -> LibraryPaths:
    values = dict(prefs or {})
    env = environ if environ is not None else os.environ
    data_root = Path(settings.db_dir).expanduser().resolve().parent
    pdf_dir, pdf_source = _resolved_path(
        preference=values.get("pdf_dir"),
        environment=env.get("KB_PDF_DIR"),
        fallback=data_root / "pdfs",
    )
    md_dir, md_source = _resolved_path(
        preference=values.get("md_dir"),
        environment=env.get("KB_MD_DIR"),
        fallback=data_root / "md_output",
    )
    return LibraryPaths(
        pdf_dir=pdf_dir,
        md_dir=md_dir,
        pdf_source=pdf_source,
        md_source=md_source,
    )
