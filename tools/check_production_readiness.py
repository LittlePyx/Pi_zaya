from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from ipaddress import ip_address
from typing import Any


def join_api_url(base_url: str, path: str) -> str:
    base = str(base_url or "").strip().rstrip("/")
    clean_path = "/" + str(path or "").strip().lstrip("/")
    return base + clean_path


def env_access_token() -> str:
    return str(os.environ.get("KB_ACCESS_TOKEN") or os.environ.get("KB_API_TOKEN") or os.environ.get("KB_AUTH_TOKEN") or "").strip()


def is_local_base_url(base_url: str) -> bool:
    try:
        parsed = urllib.parse.urlsplit(str(base_url or "").strip())
    except Exception:
        return False
    host = str(parsed.hostname or "").strip().lower()
    if host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}:
        return True
    try:
        return ip_address(host).is_loopback
    except Exception:
        return False


def resolve_access_token(base_url: str, explicit_token: str | None = None) -> str:
    if explicit_token is not None:
        return str(explicit_token or "").strip()
    if is_local_base_url(base_url):
        return env_access_token()
    return ""


def exit_code_for_status(status: str, *, allow_warning: bool = False) -> int:
    clean = str(status or "").strip().lower()
    if clean == "ok":
        return 0
    if clean == "warning":
        return 0 if allow_warning else 1
    return 2


def request_json(url: str, *, token: str = "", timeout_s: float = 8.0) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    clean_token = str(token or "").strip()
    if clean_token:
        headers["X-KB-Access-Token"] = clean_token
        headers["Authorization"] = f"Bearer {clean_token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", "replace")
    data = json.loads(raw)
    return data if isinstance(data, dict) else {}


def format_readiness(payload: dict[str, Any]) -> str:
    status = str(payload.get("status") or "error").upper()
    env = str(payload.get("env") or "unknown")
    production = "production" if payload.get("production") else "non-production"
    lines = [f"Pi-zaya readiness: {status} ({env}, {production})"]
    for item in list(payload.get("items") or []):
        if not isinstance(item, dict):
            continue
        severity = str(item.get("severity") or item.get("status") or "unknown").upper()
        label = str(item.get("label") or item.get("key") or "check")
        detail = str(item.get("detail") or "").strip()
        action = str(item.get("action") or "").strip()
        suffix = f" - {detail}" if detail else ""
        if action:
            suffix += f" [{action}]"
        lines.append(f"[{severity}] {label}{suffix}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check Pi-zaya production readiness via the FastAPI service.")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("KB_READINESS_BASE_URL") or os.environ.get("KB_BASE_URL") or "http://127.0.0.1:8000",
        help="FastAPI base URL, default: %(default)s",
    )
    parser.add_argument(
        "--token",
        default=None,
        help=(
            "Access token for protected /api/readiness. If omitted, KB_ACCESS_TOKEN/KB_API_TOKEN/KB_AUTH_TOKEN "
            "is used only for localhost/loopback base URLs."
        ),
    )
    parser.add_argument("--timeout", type=float, default=8.0, help="HTTP timeout in seconds.")
    parser.add_argument("--json", action="store_true", help="Print raw readiness JSON.")
    parser.add_argument("--allow-warning", action="store_true", help="Exit 0 when readiness status is warning.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    readiness_url = join_api_url(args.base_url, "/api/readiness")
    token = resolve_access_token(args.base_url, explicit_token=args.token)
    try:
        payload = request_json(readiness_url, token=token, timeout_s=float(args.timeout))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace") if exc.fp else str(exc)
        print(f"Readiness request failed: HTTP {exc.code} {detail}", file=sys.stderr)
        if exc.code == 401:
            print(
                "Hint: pass --token for private instances, or set KB_ENABLE_AUTH_GATE=0 and KB_REQUIRE_AUTH=0 for public deployments.",
                file=sys.stderr,
            )
        return 2
    except Exception as exc:
        print(f"Readiness request failed: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(format_readiness(payload))
    return exit_code_for_status(str(payload.get("status") or "error"), allow_warning=bool(args.allow_warning))


if __name__ == "__main__":
    raise SystemExit(main())
