from __future__ import annotations

from pathlib import Path

import pytest

from kb.test_isolation import enforce_isolated_test_paths


def _isolated_env(root: Path) -> dict[str, str]:
    return {
        "KB_TEST_ISOLATED": "1",
        "KB_TEST_DATA_ROOT": str(root),
        "KB_DB_DIR": str(root / "db"),
        "KB_CHAT_DB": str(root / "chat.sqlite3"),
        "KB_LIBRARY_DB": str(root / "library.sqlite3"),
        "KB_USER_ISSUES_DB": str(root / "user_issues.sqlite3"),
        "KB_USER_PREFS_PATH": str(root / "user_prefs.json"),
        "KB_PDF_DIR": str(root / "pdfs"),
        "KB_MD_DIR": str(root / "markdown"),
    }


def test_isolated_test_paths_accept_only_paths_inside_test_root(tmp_path: Path) -> None:
    assert enforce_isolated_test_paths(_isolated_env(tmp_path)) == tmp_path.resolve()


def test_isolated_test_paths_reject_live_chat_database(tmp_path: Path) -> None:
    env = _isolated_env(tmp_path)
    env["KB_CHAT_DB"] = str(tmp_path.parent / "chat.sqlite3")
    with pytest.raises(RuntimeError, match="KB_CHAT_DB"):
        enforce_isolated_test_paths(env)


def test_isolated_test_mode_requires_every_mutable_path(tmp_path: Path) -> None:
    env = _isolated_env(tmp_path)
    del env["KB_MD_DIR"]
    with pytest.raises(RuntimeError, match="missing: KB_MD_DIR"):
        enforce_isolated_test_paths(env)
