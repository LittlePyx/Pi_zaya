from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from fastapi import APIRouter

router = APIRouter(prefix="/api/app", tags=["app"])

_UPDATE_CACHE: dict[str, Any] = {}
_UPDATE_CACHE_LOCK = threading.Lock()


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_git(args: list[str]) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(_repo_root()),
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except Exception:
        return ""
    return completed.stdout.strip()


def _current_build_info() -> dict[str, Any]:
    commit = (os.environ.get("KB_BUILD_COMMIT") or _run_git(["rev-parse", "--short=12", "HEAD"]) or "").strip()
    exact_tag = _run_git(["describe", "--tags", "--exact-match"])
    nearest_tag = _run_git(["describe", "--tags", "--abbrev=0"])
    env_version = str(os.environ.get("KB_APP_VERSION") or "").strip()
    if env_version:
        version = env_version
        source = "env"
    elif exact_tag:
        version = exact_tag
        source = "git_tag"
    elif nearest_tag:
        version = nearest_tag
        source = "nearest_git_tag"
    else:
        version = commit or "unknown"
        source = "git_commit" if commit else "unknown"
    return {
        "name": "Pi_zaya",
        "version": version,
        "version_source": source,
        "commit": commit,
        "build_time": str(os.environ.get("KB_BUILD_TIME") or "").strip(),
        "repository": _update_repo_slug(),
    }


def _update_repo_slug() -> str:
    raw = str(os.environ.get("KB_UPDATE_REPO") or "").strip()
    if raw:
        return raw
    remote = _run_git(["remote", "get-url", "origin"])
    match = re.search(r"github\.com[:/](?P<slug>[^/\s]+/[^/\s.]+)(?:\.git)?$", remote)
    return match.group("slug") if match else "LittlePyx/Pi_zaya"


def _update_check_ttl_s() -> int:
    try:
        return max(60, int(float(os.environ.get("KB_UPDATE_CHECK_TTL_S") or "3600")))
    except Exception:
        return 3600


def _update_timeout_s() -> float:
    try:
        return max(0.5, min(15.0, float(os.environ.get("KB_UPDATE_CHECK_TIMEOUT_S") or "3")))
    except Exception:
        return 3.0


def _normalize_tag(value: object) -> str:
    return str(value or "").strip()


def _semver_key(tag: str) -> tuple[int, int, int, str] | None:
    clean = _normalize_tag(tag).lstrip("vV")
    match = re.match(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?([-.+].*)?$", clean)
    if not match:
        return None
    suffix = match.group(4) or ""
    return (
        int(match.group(1)),
        int(match.group(2) or 0),
        int(match.group(3) or 0),
        suffix,
    )


def _tag_is_newer(latest: str, current: str) -> bool | None:
    latest_clean = _normalize_tag(latest)
    current_clean = _normalize_tag(current)
    if not latest_clean or not current_clean or current_clean == "unknown":
        return None
    if latest_clean.lower() == current_clean.lower():
        return False
    latest_key = _semver_key(latest_clean)
    current_key = _semver_key(current_clean)
    if latest_key and current_key:
        return latest_key[:3] > current_key[:3]
    return None


def _fetch_latest_release(repo: str, timeout_s: float) -> dict[str, Any]:
    url = f"https://api.github.com/repos/{repo}/releases/latest"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "Pi-zaya-update-check",
        },
    )
    token = str(os.environ.get("KB_UPDATE_GITHUB_TOKEN") or os.environ.get("GITHUB_TOKEN") or "").strip()
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        raw = response.read(1024 * 512)
    data = json.loads(raw.decode("utf-8", "replace"))
    return data if isinstance(data, dict) else {}


def _public_release_payload(data: dict[str, Any]) -> dict[str, Any]:
    body = str(data.get("body") or "").strip()
    if len(body) > 2000:
        body = body[:2000].rstrip() + "\n..."
    return {
        "tag_name": str(data.get("tag_name") or "").strip(),
        "name": str(data.get("name") or data.get("tag_name") or "").strip(),
        "html_url": str(data.get("html_url") or "").strip(),
        "published_at": str(data.get("published_at") or "").strip(),
        "body": body,
        "prerelease": bool(data.get("prerelease")),
    }


def _update_instructions() -> list[str]:
    return [
        "git pull --ff-only",
        "pip install -r requirements.txt",
        "cd web",
        "npm ci",
        "npm run build",
        "cd ..",
        ".\\run_new.ps1 -StopExisting",
    ]


def _http_header(exc: urllib.error.HTTPError, name: str) -> str:
    headers = getattr(exc, "headers", None)
    if not headers:
        return ""
    try:
        return str(headers.get(name) or "").strip()
    except Exception:
        return ""


def _http_update_error_detail(exc: urllib.error.HTTPError) -> str:
    if exc.code == 404:
        return "No GitHub release was found."
    if exc.code == 403:
        remaining = _http_header(exc, "X-RateLimit-Remaining")
        if remaining == "0":
            reset_note = ""
            reset_ts = _http_rate_limit_reset_ts(exc)
            if reset_ts is not None:
                try:
                    reset_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(reset_ts))
                    reset_note = f" Try again after {reset_at}."
                except Exception:
                    reset_note = ""
            return "GitHub API rate limit reached. Set KB_UPDATE_GITHUB_TOKEN or GITHUB_TOKEN to raise the limit." + reset_note
        return "GitHub rejected the update check. If the repository is private, set KB_UPDATE_GITHUB_TOKEN."
    return f"GitHub returned HTTP {exc.code}."


def _http_rate_limit_reset_ts(exc: urllib.error.HTTPError) -> float | None:
    if exc.code != 403:
        return None
    remaining = _http_header(exc, "X-RateLimit-Remaining")
    if remaining != "0":
        return None
    reset_raw = _http_header(exc, "X-RateLimit-Reset")
    try:
        reset_ts = float(reset_raw)
    except Exception:
        return None
    return reset_ts if reset_ts > 0 else None


def _parse_retry_after_from_error(payload: dict[str, Any]) -> float | None:
    raw = str(payload.get("error") or "")
    match = re.search(r"Try again after\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})", raw)
    if not match:
        return None
    try:
        return float(time.mktime(time.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")))
    except Exception:
        return None


def _payload_retry_after(payload: dict[str, Any]) -> float | None:
    retry_after = payload.get("retry_after")
    if isinstance(retry_after, (int, float)) and retry_after > 0:
        return float(retry_after)
    return _parse_retry_after_from_error(payload)


def _update_cache_expires_at(payload: dict[str, Any], checked_at: float, ttl_s: int) -> float:
    default_expires_at = checked_at + ttl_s
    if payload.get("status") != "unavailable":
        return default_expires_at
    retry_after = _payload_retry_after(payload)
    if isinstance(retry_after, (int, float)) and retry_after > checked_at:
        return min(default_expires_at, retry_after)
    return checked_at + min(ttl_s, 300)


def _no_cached_update_payload(build: dict[str, Any], checked_at: float) -> dict[str, Any]:
    return {
        "enabled": True,
        "status": "unknown",
        "checked_at": checked_at,
        "current": build,
        "latest": None,
        "update_available": False,
        "instructions": [],
        "error": "No cached update check is available.",
    }


def _update_check_payload(*, force_refresh: bool = False, cache_only: bool = False) -> dict[str, Any]:
    build = _current_build_info()
    repo = str(build.get("repository") or _update_repo_slug())
    enabled = _truthy_env("KB_UPDATE_CHECK_ENABLED", default=True)
    checked_at = time.time()
    if not enabled:
        return {
            "enabled": False,
            "status": "disabled",
            "checked_at": checked_at,
            "current": build,
            "latest": None,
            "update_available": False,
            "instructions": [],
            "error": "",
        }

    cache_key = f"latest-release:{repo}"
    ttl_s = _update_check_ttl_s()
    with _UPDATE_CACHE_LOCK:
        cached = _UPDATE_CACHE.get(cache_key)
        cached_checked_at = float(cached.get("checked_at") or 0.0) if isinstance(cached, dict) else 0.0
        cached_payload = dict(cached.get("payload") or {}) if isinstance(cached, dict) else {}
        legacy_retry_after = _payload_retry_after(cached_payload)
        if legacy_retry_after is not None and "retry_after" not in cached_payload:
            cached_payload["retry_after"] = legacy_retry_after
        cached_expires_at = (
            float(cached.get("expires_at") or _update_cache_expires_at(cached_payload, cached_checked_at, ttl_s))
            if isinstance(cached, dict)
            else 0.0
        )
        if (
            not force_refresh
            and isinstance(cached, dict)
            and checked_at < cached_expires_at
        ):
            return dict(cached_payload)

    if cache_only and not force_refresh:
        return _no_cached_update_payload(build, checked_at)

    try:
        latest_raw = _fetch_latest_release(repo, timeout_s=_update_timeout_s())
        latest = _public_release_payload(latest_raw)
        latest_tag = str(latest.get("tag_name") or "")
        compare = _tag_is_newer(latest_tag, str(build.get("version") or ""))
        status = "ok" if compare is not None else "unknown"
        payload = {
            "enabled": True,
            "status": status,
            "checked_at": checked_at,
            "current": build,
            "latest": latest,
            "update_available": bool(compare),
            "instructions": _update_instructions() if bool(compare) else [],
            "error": "" if status == "ok" else "Current build is not comparable to the latest release tag.",
        }
    except urllib.error.HTTPError as exc:
        detail = _http_update_error_detail(exc)
        retry_after = _http_rate_limit_reset_ts(exc)
        payload = {
            "enabled": True,
            "status": "unavailable",
            "checked_at": checked_at,
            "current": build,
            "latest": None,
            "update_available": False,
            "instructions": [],
            "error": detail,
        }
        if retry_after is not None:
            payload["retry_after"] = retry_after
    except Exception as exc:
        payload = {
            "enabled": True,
            "status": "unavailable",
            "checked_at": checked_at,
            "current": build,
            "latest": None,
            "update_available": False,
            "instructions": [],
            "error": str(exc),
        }

    with _UPDATE_CACHE_LOCK:
        _UPDATE_CACHE[cache_key] = {
            "checked_at": checked_at,
            "expires_at": _update_cache_expires_at(payload, checked_at, ttl_s),
            "payload": dict(payload),
        }
    return payload


@router.get("/version")
def get_app_version():
    return _current_build_info()


@router.get("/update-check")
def get_update_check(refresh: bool = False, cache_only: bool = False):
    return _update_check_payload(force_refresh=bool(refresh), cache_only=bool(cache_only))
