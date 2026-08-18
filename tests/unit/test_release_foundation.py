from __future__ import annotations

import json
from pathlib import Path

from kb.app_paths import configure_release_environment, release_data_root
from kb.version import read_app_version, release_tag


def test_canonical_version_is_valid_and_tagged() -> None:
    version = read_app_version()

    assert version == "0.1.0-beta.9"
    assert release_tag(version) == "v0.1.0-beta.9"


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
    builder_path = root / "tools" / "release" / "build_windows_portable.ps1"
    smoke_path = root / "tools" / "release" / "smoke_windows_portable.ps1"
    installer_builder_path = root / "tools" / "release" / "build_windows_installer.ps1"
    installer_smoke_path = root / "tools" / "release" / "smoke_windows_installer.ps1"
    installer_script_path = root / "packaging" / "windows" / "Pi_zaya.iss"
    native_launcher_path = root / "packaging" / "windows" / "PiZayaLauncher.cs"
    builder = builder_path.read_text(encoding="utf-8")
    smoke = smoke_path.read_text(encoding="utf-8")
    installer_builder = installer_builder_path.read_text(encoding="utf-8")
    installer_smoke = installer_smoke_path.read_text(encoding="utf-8")
    installer_script = installer_script_path.read_text(encoding="utf-8")
    launcher = (root / "packaging" / "windows" / "Start-Pi-zaya.ps1").read_text(encoding="utf-8")
    stopper = (root / "packaging" / "windows" / "Stop-Pi-zaya.ps1").read_text(encoding="utf-8")
    native_launcher = native_launcher_path.read_text(encoding="utf-8")
    chinese_readme_path = root / "packaging" / "windows" / "README-中文.md"
    chinese_readme = chinese_readme_path.read_text(encoding="utf-8")
    ignore_lines = (root / ".gitignore").read_text(encoding="utf-8").splitlines()

    assert builder_path.is_file()
    assert smoke_path.is_file()
    assert installer_builder_path.is_file()
    assert installer_smoke_path.is_file()
    assert installer_script_path.is_file()
    assert native_launcher_path.is_file()
    assert "/release/" in ignore_lines
    assert "release/" not in ignore_lines
    assert "tags:" in workflow
    assert "LICENSE is required" in workflow
    assert "smoke_windows_portable.ps1" in workflow
    assert "build_windows_installer.ps1" in workflow
    assert "smoke_windows_installer.ps1" in workflow
    assert "WINDOWS_SIGNING_CERT_BASE64" in workflow
    assert "Pyrsys B\\.V\\." in workflow
    assert "--prerelease" in workflow
    assert "PythonRuntime Embedded" in workflow
    assert "requirements-file: requirements-release.txt" in workflow
    assert 'PYTHONUTF8: "1"' in workflow
    assert 'PYTHONIOENCODING: "utf-8"' in workflow
    assert "Retain frontend E2E failure diagnostics" in workflow
    assert "AllowMissingLicense" in builder
    assert "AllowDirty" in builder
    assert "source_dirty" in builder
    assert chinese_readme_path.is_file()
    assert '"packaging\\windows\\README-中文.md"' in builder
    assert "Build-WindowsLauncher" in builder
    assert "Invoke-PiZayaAuthenticodeSign" in builder
    assert "launcher_signed" in builder
    assert "AssemblyInformationalVersion" in builder
    assert '"Pi_zaya.exe"' in builder
    assert 'entrypoint = "Pi_zaya.exe"' in builder
    assert 'fallback_entrypoint = "Start-Pi-zaya.cmd"' in builder
    assert '"README-中文.md"' in smoke
    assert '"Pi_zaya.exe"' in smoke
    assert "WaitForExit(65000)" in smoke
    assert "occupied preferred port" in smoke
    assert "配置 API Key 和模型" in chinese_readme
    assert "不会被发送给多家服务" in chinese_readme
    assert "不会一直卡在加载状态" in chinese_readme
    assert "系统托盘" in chinese_readme
    assert "Pi_zaya.exe" in chinese_readme
    assert '"LICENSE"' in smoke
    assert 'manifest.license -ne "MIT"' in smoke
    assert "manifest.source_dirty" in smoke
    assert "AllowDirty" in smoke
    assert "ArchivePath" in smoke
    assert "CleanProfile" in smoke
    assert "Expand-Archive" in smoke
    assert '"python.exe", "python3.exe", "node.exe", "npm.cmd"' in smoke
    assert "Clean-profile launch did not use the expected isolated data directory" in smoke
    assert "Smoke-test downloaded ZIP on a clean Windows profile" in workflow
    assert "-ArchivePath" in workflow
    assert "-CleanProfile" in workflow
    assert "Copyright \\(c\\) [^\\r\\n]+ LittlePyx" in workflow
    assert 'KB_RELEASE_MODE = "1"' in launcher
    assert 'KB_SERVER_HOST = "127.0.0.1"' in launcher
    assert "Get-AvailableLoopbackPort" in launcher
    assert "Elapsed.TotalSeconds -lt 45" in launcher
    assert "streamlit run" not in launcher.lower()
    assert "did not stop within 10 seconds" in stopper
    assert "NotifyIcon" in native_launcher
    assert "MutexName" in native_launcher
    assert "StartTimeoutMilliseconds = 60000" in native_launcher
    assert "Do not call the unbounded WaitForExit()" in native_launcher
    assert "PrivilegesRequired=lowest" in installer_script
    assert "AppMutex=Local\\Pi_zaya.WindowsLauncher" in installer_script
    assert "SignedUninstaller=yes" in installer_script
    assert "uninstall_preserves_user_data" in installer_builder
    assert "Get-PiZayaAuthenticodeState" in installer_builder
    assert "In-place installer upgrade removed user data" in installer_smoke
    assert "Uninstaller removed Pi_zaya user data" in installer_smoke
    assert "A real Pi_zaya installation already exists" in installer_smoke


def test_browser_gates_retain_first_failure_diagnostics() -> None:
    root = Path(__file__).resolve().parents[2]
    config = (root / "web" / "playwright.config.ts").read_text(encoding="utf-8")
    ci_workflow = (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    release_workflow = (root / ".github" / "workflows" / "release-windows.yml").read_text(encoding="utf-8")

    assert "trace: 'retain-on-failure'" in config
    assert "screenshot: 'only-on-failure'" in config
    assert "Retain frontend E2E failure diagnostics" in ci_workflow
    assert "Retain frontend E2E failure diagnostics" in release_workflow
    assert "web/test-results/" in ci_workflow
    assert "web/test-results/" in release_workflow


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
