from __future__ import annotations

from email.message import Message
import urllib.error

from fastapi.testclient import TestClient

from api.routers import app as app_router


def _release(tag: str = "v1.2.0") -> dict:
    return {
        "tag_name": tag,
        "name": f"Release {tag}",
        "html_url": f"https://github.com/LittlePyx/Pi_zaya/releases/tag/{tag}",
        "published_at": "2026-06-08T00:00:00Z",
        "body": "Release notes",
        "prerelease": False,
    }


def _build(version: str = "v1.1.0") -> dict:
    return {
        "name": "Pi_zaya",
        "version": version,
        "version_source": "env",
        "commit": "abc123",
        "build_time": "",
        "repository": "LittlePyx/Pi_zaya",
    }


def test_app_version_payload_uses_current_build_info(monkeypatch):
    monkeypatch.setattr(app_router, "_current_build_info", lambda: _build("v2.0.0"))

    payload = app_router.get_app_version()

    assert payload["version"] == "v2.0.0"
    assert payload["repository"] == "LittlePyx/Pi_zaya"


def test_update_check_reports_newer_latest_release(monkeypatch):
    app_router._UPDATE_CACHE.clear()
    monkeypatch.setenv("KB_UPDATE_CHECK_ENABLED", "1")
    monkeypatch.setattr(app_router, "_current_build_info", lambda: _build("v1.1.0"))
    monkeypatch.setattr(app_router, "_fetch_latest_release", lambda repo, timeout_s: _release("v1.2.0"))

    payload = app_router.get_update_check(refresh=True)

    assert payload["enabled"] is True
    assert payload["status"] == "ok"
    assert payload["current"]["version"] == "v1.1.0"
    assert payload["latest"]["tag_name"] == "v1.2.0"
    assert payload["update_available"] is True
    assert "git pull --ff-only" in payload["instructions"]


def test_update_check_reports_up_to_date(monkeypatch):
    app_router._UPDATE_CACHE.clear()
    monkeypatch.setenv("KB_UPDATE_CHECK_ENABLED", "1")
    monkeypatch.setattr(app_router, "_current_build_info", lambda: _build("v1.2.0"))
    monkeypatch.setattr(app_router, "_fetch_latest_release", lambda repo, timeout_s: _release("v1.2.0"))

    payload = app_router.get_update_check(refresh=True)

    assert payload["status"] == "ok"
    assert payload["update_available"] is False
    assert payload["instructions"] == []


def test_update_check_can_be_disabled(monkeypatch):
    app_router._UPDATE_CACHE.clear()
    monkeypatch.setenv("KB_UPDATE_CHECK_ENABLED", "0")
    monkeypatch.setattr(app_router, "_current_build_info", lambda: _build("v1.1.0"))

    payload = app_router.get_update_check(refresh=True)

    assert payload["enabled"] is False
    assert payload["status"] == "disabled"
    assert payload["update_available"] is False


def test_update_check_returns_unavailable_instead_of_raising(monkeypatch):
    app_router._UPDATE_CACHE.clear()
    monkeypatch.setenv("KB_UPDATE_CHECK_ENABLED", "1")
    monkeypatch.setattr(app_router, "_current_build_info", lambda: _build("v1.1.0"))

    def fail(repo: str, timeout_s: float) -> dict:
        raise urllib.error.HTTPError(
            url="https://api.github.com/repos/LittlePyx/Pi_zaya/releases/latest",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(app_router, "_fetch_latest_release", fail)

    payload = app_router.get_update_check(refresh=True)

    assert payload["status"] == "unavailable"
    assert payload["update_available"] is False
    assert "release" in payload["error"].lower()


def test_update_check_explains_github_rate_limit(monkeypatch):
    app_router._UPDATE_CACHE.clear()
    monkeypatch.setenv("KB_UPDATE_CHECK_ENABLED", "1")
    monkeypatch.setattr(app_router, "_current_build_info", lambda: _build("v1.1.0"))
    headers = Message()
    headers["X-RateLimit-Remaining"] = "0"
    headers["X-RateLimit-Reset"] = "1780979216"

    def fail(repo: str, timeout_s: float) -> dict:
        raise urllib.error.HTTPError(
            url="https://api.github.com/repos/LittlePyx/Pi_zaya/releases/latest",
            code=403,
            msg="Forbidden",
            hdrs=headers,
            fp=None,
        )

    monkeypatch.setattr(app_router, "_fetch_latest_release", fail)

    payload = app_router.get_update_check(refresh=True)

    assert payload["status"] == "unavailable"
    assert payload["update_available"] is False
    assert "rate limit" in payload["error"].lower()
    assert "KB_UPDATE_GITHUB_TOKEN" in payload["error"]
    assert payload["retry_after"] == 1780979216.0


def test_update_check_handles_github_403_without_headers(monkeypatch):
    app_router._UPDATE_CACHE.clear()
    monkeypatch.setenv("KB_UPDATE_CHECK_ENABLED", "1")
    monkeypatch.setattr(app_router, "_current_build_info", lambda: _build("v1.1.0"))

    def fail(repo: str, timeout_s: float) -> dict:
        raise urllib.error.HTTPError(
            url="https://api.github.com/repos/LittlePyx/Pi_zaya/releases/latest",
            code=403,
            msg="Forbidden",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(app_router, "_fetch_latest_release", fail)

    payload = app_router.get_update_check(refresh=True)

    assert payload["status"] == "unavailable"
    assert "private" in payload["error"].lower()
    assert "retry_after" not in payload


def test_update_check_uses_cache(monkeypatch):
    app_router._UPDATE_CACHE.clear()
    calls = {"count": 0}
    monkeypatch.setenv("KB_UPDATE_CHECK_ENABLED", "1")
    monkeypatch.setattr(app_router, "_current_build_info", lambda: _build("v1.1.0"))

    def fetch(repo: str, timeout_s: float) -> dict:
        calls["count"] += 1
        return _release("v1.2.0")

    monkeypatch.setattr(app_router, "_fetch_latest_release", fetch)

    first = app_router.get_update_check(refresh=True)
    second = app_router.get_update_check(refresh=False)

    assert first["update_available"] is True
    assert second["update_available"] is True
    assert calls["count"] == 1


def test_rate_limit_cache_expires_at_github_reset(monkeypatch):
    app_router._UPDATE_CACHE.clear()
    calls = {"count": 0}
    now = {"value": 1000.0}
    monkeypatch.setenv("KB_UPDATE_CHECK_ENABLED", "1")
    monkeypatch.setenv("KB_UPDATE_CHECK_TTL_S", "3600")
    monkeypatch.setattr(app_router.time, "time", lambda: now["value"])
    monkeypatch.setattr(app_router, "_current_build_info", lambda: _build("v1.1.0"))
    headers = Message()
    headers["X-RateLimit-Remaining"] = "0"
    headers["X-RateLimit-Reset"] = "1030"

    def fetch(repo: str, timeout_s: float) -> dict:
        calls["count"] += 1
        if calls["count"] == 1:
            raise urllib.error.HTTPError(
                url="https://api.github.com/repos/LittlePyx/Pi_zaya/releases/latest",
                code=403,
                msg="Forbidden",
                hdrs=headers,
                fp=None,
            )
        return _release("v1.2.0")

    monkeypatch.setattr(app_router, "_fetch_latest_release", fetch)

    first = app_router.get_update_check(refresh=True)
    now["value"] = 1010.0
    second = app_router.get_update_check(refresh=False)
    now["value"] = 1031.0
    third = app_router.get_update_check(refresh=False)

    assert first["status"] == "unavailable"
    assert second["status"] == "unavailable"
    assert third["status"] == "ok"
    assert third["update_available"] is True
    assert calls["count"] == 2


def test_legacy_rate_limit_cache_expires_at_error_reset(monkeypatch):
    app_router._UPDATE_CACHE.clear()
    calls = {"count": 0}
    now = {"value": 1000.0}
    retry_after = 1030.0
    retry_at = app_router.time.strftime("%Y-%m-%d %H:%M:%S", app_router.time.localtime(retry_after))
    monkeypatch.setenv("KB_UPDATE_CHECK_ENABLED", "1")
    monkeypatch.setenv("KB_UPDATE_CHECK_TTL_S", "3600")
    monkeypatch.setattr(app_router.time, "time", lambda: now["value"])
    monkeypatch.setattr(app_router, "_current_build_info", lambda: _build("v1.1.0"))

    app_router._UPDATE_CACHE["latest-release:LittlePyx/Pi_zaya"] = {
        "checked_at": 1000.0,
        "payload": {
            "enabled": True,
            "status": "unavailable",
            "checked_at": 1000.0,
            "current": _build("v1.1.0"),
            "latest": None,
            "update_available": False,
            "instructions": [],
            "error": "GitHub API rate limit reached. Set KB_UPDATE_GITHUB_TOKEN or GITHUB_TOKEN to raise the limit. "
            f"Try again after {retry_at}.",
        },
    }

    def fetch(repo: str, timeout_s: float) -> dict:
        calls["count"] += 1
        return _release("v1.2.0")

    monkeypatch.setattr(app_router, "_fetch_latest_release", fetch)

    cached = app_router.get_update_check(refresh=False)
    now["value"] = 1031.0
    refreshed = app_router.get_update_check(refresh=False)

    assert cached["status"] == "unavailable"
    assert cached["retry_after"] == retry_after
    assert refreshed["status"] == "ok"
    assert refreshed["update_available"] is True
    assert calls["count"] == 1


def test_update_check_cache_only_does_not_fetch_without_cache(monkeypatch):
    app_router._UPDATE_CACHE.clear()
    calls = {"count": 0}
    monkeypatch.setenv("KB_UPDATE_CHECK_ENABLED", "1")
    monkeypatch.setattr(app_router, "_current_build_info", lambda: _build("v1.1.0"))

    def fetch(repo: str, timeout_s: float) -> dict:
        calls["count"] += 1
        return _release("v1.2.0")

    monkeypatch.setattr(app_router, "_fetch_latest_release", fetch)

    payload = app_router.get_update_check(cache_only=True)

    assert payload["status"] == "unknown"
    assert payload["update_available"] is False
    assert "cached" in payload["error"].lower()
    assert calls["count"] == 0


def test_update_check_cache_only_reuses_fresh_cache(monkeypatch):
    app_router._UPDATE_CACHE.clear()
    calls = {"count": 0}
    monkeypatch.setenv("KB_UPDATE_CHECK_ENABLED", "1")
    monkeypatch.setattr(app_router, "_current_build_info", lambda: _build("v1.1.0"))

    def fetch(repo: str, timeout_s: float) -> dict:
        calls["count"] += 1
        return _release("v1.2.0")

    monkeypatch.setattr(app_router, "_fetch_latest_release", fetch)

    first = app_router.get_update_check(refresh=True)
    second = app_router.get_update_check(cache_only=True)

    assert first["update_available"] is True
    assert second["update_available"] is True
    assert calls["count"] == 1


def test_update_check_route_is_mounted(monkeypatch):
    from api.main import app

    app_router._UPDATE_CACHE.clear()
    monkeypatch.setenv("KB_UPDATE_CHECK_ENABLED", "1")
    monkeypatch.setattr(app_router, "_current_build_info", lambda: _build("v1.1.0"))
    monkeypatch.setattr(app_router, "_fetch_latest_release", lambda repo, timeout_s: _release("v1.2.0"))

    response = TestClient(app).get("/api/app/update-check?refresh=true")

    assert response.status_code == 200
    payload = response.json()
    assert payload["update_available"] is True
    assert payload["latest"]["tag_name"] == "v1.2.0"
