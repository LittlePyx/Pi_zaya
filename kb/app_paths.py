from __future__ import annotations

import os
from collections.abc import MutableMapping
from pathlib import Path


_TRUE_VALUES = {"1", "true", "yes", "on"}


def _enabled(value: object) -> bool:
    return str(value or "").strip().lower() in _TRUE_VALUES


def release_data_root(env: MutableMapping[str, str] | None = None) -> Path | None:
    """Resolve the data root only for an explicitly requested packaged runtime."""

    target = env if env is not None else os.environ
    configured = str(target.get("KB_APP_DATA_DIR") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    if not _enabled(target.get("KB_RELEASE_MODE")):
        return None
    local_app_data = str(target.get("LOCALAPPDATA") or "").strip()
    if local_app_data:
        return (Path(local_app_data).expanduser() / "Pi_zaya").resolve()
    return (Path.home() / ".pi_zaya").resolve()


def release_path_defaults(root: Path) -> dict[str, str]:
    data_root = Path(root).expanduser().resolve()
    return {
        "KB_DB_DIR": str(data_root / "db"),
        "KB_CHAT_DB": str(data_root / "chat.sqlite3"),
        "KB_LIBRARY_DB": str(data_root / "library.sqlite3"),
        "KB_USER_ISSUES_DB": str(data_root / "user_issues.sqlite3"),
        "KB_USER_PREFS_PATH": str(data_root / "user_prefs.json"),
        "KB_PDF_DIR": str(data_root / "pdfs"),
        "KB_MD_DIR": str(data_root / "markdown"),
        "KB_BACKUP_DIR": str(data_root / "backups"),
        "KB_DIAGNOSTICS_DIR": str(data_root / "diagnostics"),
        "KB_RESTORE_AUDIT_PATH": str(data_root / "restore_audit.jsonl"),
    }


def configure_release_environment(
    env: MutableMapping[str, str] | None = None,
    *,
    create_directories: bool = False,
) -> Path | None:
    """Apply release-only data defaults without overriding operator choices."""

    target = env if env is not None else os.environ
    root = release_data_root(target)
    if root is None:
        return None
    target.setdefault("KB_APP_DATA_DIR", str(root))
    for key, value in release_path_defaults(root).items():
        target.setdefault(key, value)
    if create_directories:
        for folder in (
            root,
            Path(target["KB_DB_DIR"]),
            Path(target["KB_PDF_DIR"]),
            Path(target["KB_MD_DIR"]),
            Path(target["KB_BACKUP_DIR"]),
            Path(target["KB_DIAGNOSTICS_DIR"]),
            root / "logs",
            root / "runtime",
        ):
            folder.mkdir(parents=True, exist_ok=True)
    return root
