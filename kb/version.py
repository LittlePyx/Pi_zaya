from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VERSION_FILE = PROJECT_ROOT / "VERSION"
_VERSION_RE = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
    r"(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
)


def read_app_version(path: Path | None = None) -> str:
    """Read and validate the repository's canonical application version."""

    target = Path(path) if path is not None else VERSION_FILE
    try:
        version = target.read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"
    return version if _VERSION_RE.fullmatch(version) else "unknown"


def release_tag(version: str | None = None) -> str:
    resolved = str(version or read_app_version()).strip()
    return f"v{resolved}" if resolved and resolved != "unknown" else "unknown"
