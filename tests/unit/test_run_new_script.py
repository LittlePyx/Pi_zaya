from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_run_new_starts_background_helpers_hidden() -> None:
    script = (ROOT / "run_new.ps1").read_text(encoding="utf-8")
    start_blocks = re.findall(r"Start-Process `.*?(?=\n\n)", script, flags=re.DOTALL)
    backend_block = next(block for block in start_blocks if "$pythonExe" in block)
    frontend_block = next(block for block in start_blocks if "$npmExe" in block)

    assert "-WindowStyle Hidden" in backend_block
    assert "-WindowStyle Hidden" in frontend_block
    assert "streamlit" not in script.lower()
    assert "app.py" not in script.lower()


def test_run_new_preflights_missing_dependencies_with_actionable_hints() -> None:
    script = (ROOT / "run_new.ps1").read_text(encoding="utf-8")

    assert "function Test-BackendDepsReady" in script
    assert "$code = @'" in script
    assert "required = ('fastapi', 'uvicorn')" in script
    assert "function Test-FrontendDepsReady" in script
    assert "node_modules\\.bin\\vite.cmd" in script
    assert "Backend dependencies are missing" in script
    assert "-InstallBackendDeps -InstallFrontendDeps -StopExisting" in script
    assert "Frontend dependencies are missing" in script
    assert "cd web; npm ci" in script


def test_run_new_uses_npm_ci_when_lockfile_exists() -> None:
    script = (ROOT / "run_new.ps1").read_text(encoding="utf-8")
    install_block = re.search(r"function Install-FrontendDeps.*?\n}", script, flags=re.DOTALL)

    assert install_block is not None
    text = install_block.group(0)
    assert 'package-lock.json' in text
    assert "& $NpmExe ci" in text
    assert "& $NpmExe install" in text


def test_run_new_keeps_local_user_app_public_by_default() -> None:
    script = (ROOT / "run_new.ps1").read_text(encoding="utf-8")

    assert "[switch]$AllowAuthGate" in script
    assert "Access-token gate settings detected" in script
    assert 'SetEnvironmentVariable("KB_REQUIRE_AUTH", "0", "Process")' in script
    assert 'SetEnvironmentVariable("KB_ENABLE_AUTH_GATE", "0", "Process")' in script
    assert 'SetEnvironmentVariable("KB_PRIVATE_INSTANCE_AUTH", "0", "Process")' in script
    assert 'SetEnvironmentVariable("KB_ALLOW_LOCAL_AUTH_GATE", "0", "Process")' in script
    assert 'SetEnvironmentVariable("VITE_ENABLE_AUTH_GATE", "0", "Process")' in script
    assert 'SetEnvironmentVariable("VITE_PRIVATE_INSTANCE_AUTH", "0", "Process")' in script
    assert 'SetEnvironmentVariable("VITE_ALLOW_LOCAL_AUTH_GATE", "0", "Process")' in script
    assert 'SetEnvironmentVariable("KB_ENABLE_AUTH_GATE", "1", "Process")' in script
    assert 'SetEnvironmentVariable("KB_PRIVATE_INSTANCE_AUTH", "1", "Process")' in script
    assert 'SetEnvironmentVariable("KB_ALLOW_LOCAL_AUTH_GATE", "1", "Process")' in script
    assert 'SetEnvironmentVariable("VITE_ENABLE_AUTH_GATE", "1", "Process")' in script
    assert 'SetEnvironmentVariable("VITE_PRIVATE_INSTANCE_AUTH", "1", "Process")' in script
    assert 'SetEnvironmentVariable("VITE_ALLOW_LOCAL_AUTH_GATE", "1", "Process")' in script
    assert "Use -AllowAuthGate only when testing a private access-token gate" in script
    assert "Access-token gate: OFF; users do not need a token." in script


def test_run_new_keeps_browser_api_same_origin_while_proxying_backend() -> None:
    script = (ROOT / "run_new.ps1").read_text(encoding="utf-8")
    vite_config = (ROOT / "web" / "vite.config.ts").read_text(encoding="utf-8")

    assert 'SetEnvironmentVariable("VITE_BACKEND_URL", "", "Process")' in script
    assert 'SetEnvironmentVariable("VITE_BACKEND_PROXY_TARGET", "http://$BackendHost`:$BackendPort", "Process")' in script
    assert "Local dev mode clears it so the browser uses same-origin /api" in script
    assert "process.env.VITE_BACKEND_PROXY_TARGET" in vite_config
    assert "process.env.VITE_BACKEND_URL" in vite_config
