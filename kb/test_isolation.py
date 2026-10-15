from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path


_TRUE_VALUES = {"1", "true", "yes", "on"}
_ISOLATED_PATH_KEYS = (
    "KB_DB_DIR",
    "KB_CHAT_DB",
    "KB_LIBRARY_DB",
    "KB_USER_ISSUES_DB",
    "KB_USER_PREFS_PATH",
    "KB_PDF_DIR",
    "KB_MD_DIR",
)


def isolated_test_mode_enabled(env: Mapping[str, str] | None = None) -> bool:
    values = env if env is not None else os.environ
    return str(values.get("KB_TEST_ISOLATED") or "").strip().lower() in _TRUE_VALUES


def enforce_isolated_test_paths(env: Mapping[str, str] | None = None) -> Path | None:
    """Reject a test process whose mutable data paths escape its temporary root."""

    values = env if env is not None else os.environ
    if not isolated_test_mode_enabled(values):
        return None
    raw_root = str(values.get("KB_TEST_DATA_ROOT") or "").strip()
    if not raw_root:
        raise RuntimeError("KB_TEST_ISOLATED requires KB_TEST_DATA_ROOT")
    root = Path(raw_root).expanduser().resolve()
    missing: list[str] = []
    outside: list[str] = []
    for key in _ISOLATED_PATH_KEYS:
        raw_path = str(values.get(key) or "").strip()
        if not raw_path:
            missing.append(key)
            continue
        resolved = Path(raw_path).expanduser().resolve()
        if not resolved.is_relative_to(root):
            outside.append(f"{key}={resolved}")
    if missing or outside:
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if outside:
            details.append(f"outside {root}: {', '.join(outside)}")
        raise RuntimeError("Unsafe isolated test data paths; " + "; ".join(details))
    return root
