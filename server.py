#!/usr/bin/env python3
from __future__ import annotations

import os
import uvicorn
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env", override=False)

from api.main import app


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, min_value: int, max_value: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    clean = str(raw).strip()
    try:
        value = int(clean)
    except Exception:
        raise SystemExit(f"{name} must be an integer between {min_value} and {max_value}; got {raw!r}")
    if value < min_value or value > max_value:
        raise SystemExit(f"{name} must be an integer between {min_value} and {max_value}; got {value}")
    return value


def _readiness_host_for_display(host: str) -> str:
    clean = str(host or "").strip()
    if clean in {"0.0.0.0", "::", "[::]"}:
        return "127.0.0.1"
    return clean or "127.0.0.1"


def _readiness_base_url(*, host: str, port: int) -> str:
    display_host = _readiness_host_for_display(host)
    if ":" in display_host and not (display_host.startswith("[") and display_host.endswith("]")):
        display_host = f"[{display_host}]"
    return f"http://{display_host}:{port}"


def _startup_readiness_lines(payload: dict, *, host: str = "127.0.0.1", port: int = 8000) -> list[str]:
    status = str(payload.get("status") or "error").upper()
    env = str(payload.get("env") or "unknown")
    production = "production" if payload.get("production") else "non-production"
    lines = [f"Pi-zaya startup preflight: {status} ({env}, {production})"]
    issue_count = 0
    for item in list(payload.get("items") or []):
        if not isinstance(item, dict):
            continue
        severity = str(item.get("severity") or item.get("status") or "unknown").lower()
        if severity == "ok":
            continue
        issue_count += 1
        label = str(item.get("label") or item.get("key") or "check")
        detail = str(item.get("detail") or "").strip()
        action = str(item.get("action") or "").strip()
        suffix = f" - {detail}" if detail else ""
        if action:
            suffix += f" [{action}]"
        lines.append(f"  [{severity.upper()}] {label}{suffix}")
    if issue_count == 0:
        lines.append("  All startup readiness checks passed.")
    base_url = _readiness_base_url(host=host, port=port)
    token_arg = " --token <access-token>" if bool(payload.get("auth_required")) else ""
    lines.append(
        "  Full check: python tools\\check_production_readiness.py "
        f"--base-url {base_url}{token_arg}"
    )
    return lines


def _startup_preflight_exit_code(payload: dict, *, strict: bool = False) -> int:
    if strict and str(payload.get("status") or "").lower() == "error":
        return 2
    return 0


def _run_startup_preflight(*, host: str, port: int, app_env: str) -> int:
    production = app_env in {"prod", "production"}
    if not _env_flag("KB_STARTUP_PREFLIGHT", default=production):
        return 0
    from api.deps import get_settings
    from api.routers.settings import production_readiness_payload

    payload = production_readiness_payload(get_settings())
    for line in _startup_readiness_lines(payload, host=host, port=port):
        print(line)
    strict = _env_flag("KB_STARTUP_STRICT", default=False)
    exit_code = _startup_preflight_exit_code(payload, strict=strict)
    if exit_code:
        print("Pi-zaya startup stopped because KB_STARTUP_STRICT=1 and readiness is blocked.")
    return exit_code

# Serve frontend static files in production
_DIST = Path(__file__).parent / "web" / "dist"
if _DIST.is_dir():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="static")

if __name__ == "__main__":
    host = os.environ.get("KB_SERVER_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = _env_int("KB_SERVER_PORT", 8000, min_value=1, max_value=65535)
    app_env = (os.environ.get("KB_APP_ENV") or os.environ.get("KB_ENV") or "development").strip().lower()
    reload_default = "0" if app_env in {"prod", "production"} else "1"
    reload_enabled = os.environ.get("KB_SERVER_RELOAD", reload_default).strip().lower() not in {"0", "false", "no", "off"}
    preflight_exit = _run_startup_preflight(host=host, port=port, app_env=app_env)
    if preflight_exit:
        raise SystemExit(preflight_exit)
    uvicorn.run("server:app", host=host, port=port, reload=reload_enabled)
