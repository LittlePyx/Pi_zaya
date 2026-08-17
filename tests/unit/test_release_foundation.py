from __future__ import annotations

import json
from pathlib import Path

from kb.app_paths import configure_release_environment, release_data_root
from kb.version import read_app_version, release_tag


def test_canonical_version_is_valid_and_tagged() -> None:
    version = read_app_version()

    assert version == "0.1.0-beta.3"
    assert release_tag(version) == "v0.1.0-beta.3"


def test_invalid_version_file_is_rejected(tmp_path: Path) -> None:
    version_file = tmp_path / "VERSION"
    version_file.write_text("release one\n", encoding="utf-8")

    assert read_app_version(version_file) == "unknown"


def test_development_mode_does_not_relocate_data(tmp_path: Path) -> None:
    env = {"LOCALAPPDATA": str(tmp_path)}

    assert release_data_root(env) is None
    assert configure_release_environment(env) is None
    assert "KB_DB_DIR" not in env


def test_release_mode_uses_local_app_data_and_preserves_overrides(tmp_path: Path) -> None:
    explicit_db = tmp_path / "operator-db"
    env = {
        "KB_RELEASE_MODE": "1",
        "LOCALAPPDATA": str(tmp_path),
        "KB_DB_DIR": str(explicit_db),
    }

    root = configure_release_environment(env)

    assert root == (tmp_path / "Pi_zaya").resolve()
    assert Path(env["KB_DB_DIR"]) == explicit_db
    assert Path(env["KB_CHAT_DB"]) == root / "chat.sqlite3"
    assert Path(env["KB_USER_PREFS_PATH"]) == root / "user_prefs.json"
    assert Path(env["KB_PDF_DIR"]) == root / "pdfs"
    assert Path(env["KB_MD_DIR"]) == root / "markdown"


def test_explicit_app_data_dir_enables_release_paths(tmp_path: Path) -> None:
    root = tmp_path / "portable-data"
    env = {"KB_APP_DATA_DIR": str(root)}

    resolved = configure_release_environment(env, create_directories=True)

    assert resolved == root.resolve()
    assert (root / "db").is_dir()
    assert (root / "logs").is_dir()
    assert (root / "runtime").is_dir()


def test_version_is_consistent_across_frontend_manifests() -> None:
    root = Path(__file__).resolve().parents[2]
    expected = read_app_version()
    package = json.loads((root / "web" / "package.json").read_text(encoding="utf-8"))
    lock = json.loads((root / "web" / "package-lock.json").read_text(encoding="utf-8"))

    assert package["version"] == expected
    assert package["license"] == "MIT"
    assert lock["version"] == expected
    assert lock["packages"][""]["version"] == expected
    assert lock["packages"][""]["license"] == "MIT"


def test_windows_release_contract_keeps_license_and_smoke_gates() -> None:
    root = Path(__file__).resolve().parents[2]
    workflow = (root / ".github" / "workflows" / "release-windows.yml").read_text(encoding="utf-8")
    builder = (root / "tools" / "release" / "build_windows_portable.ps1").read_text(encoding="utf-8")
    launcher = (root / "packaging" / "windows" / "Start-Pi-zaya.ps1").read_text(encoding="utf-8")

    assert "tags:" in workflow
    assert "LICENSE is required" in workflow
    assert "smoke_windows_portable.ps1" in workflow
    assert "--prerelease" in workflow
    assert "PythonRuntime Embedded" in workflow
    assert "requirements-file: requirements-release.txt" in workflow
    assert 'PYTHONUTF8: "1"' in workflow
    assert 'PYTHONIOENCODING: "utf-8"' in workflow
    assert "AllowMissingLicense" in builder
    assert "AllowDirty" in builder
    assert "source_dirty" in builder
    assert "Copyright \\(c\\) [^\\r\\n]+ LittlePyx" in workflow
    assert 'KB_RELEASE_MODE = "1"' in launcher
    assert 'KB_SERVER_HOST = "127.0.0.1"' in launcher
    assert "streamlit run" not in launcher.lower()


def test_mit_license_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    license_text = (root / "LICENSE").read_text(encoding="utf-8")

    assert license_text.startswith("MIT License\n")
    assert "Copyright (c) 2026 LittlePyx" in license_text
    assert "Permission is hereby granted, free of charge" in license_text
    assert "THE SOFTWARE IS PROVIDED \"AS IS\"" in license_text


def test_release_runtime_dependencies_are_exactly_pinned() -> None:
    root = Path(__file__).resolve().parents[2]
    lines = (root / "requirements-release.txt").read_text(encoding="utf-8").splitlines()
    requirements = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]

    assert requirements
    assert all("==" in requirement for requirement in requirements)
    assert "openai==2.16.0" in requirements
    assert "PyMuPDF==1.26.5" in requirements
    assert "pdfplumber==0.11.8" in requirements
