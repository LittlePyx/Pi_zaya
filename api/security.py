from __future__ import annotations

import hashlib
import secrets

from fastapi import Request

from kb.config import Settings, load_settings

AUTH_COOKIE_NAME = "kb_access_token"
AUTH_HEADER_NAME = "X-KB-Access-Token"


def auth_settings() -> Settings:
    return load_settings()


def auth_token_configured(settings: Settings | None = None) -> bool:
    s = settings or auth_settings()
    return bool(getattr(s, "access_token", None) or getattr(s, "access_token_sha256", None))


def access_token_valid(token: str | None, settings: Settings | None = None) -> bool:
    raw = str(token or "").strip()
    if not raw:
        return False
    s = settings or auth_settings()
    expected = str(getattr(s, "access_token", None) or "").strip()
    if expected and secrets.compare_digest(raw, expected):
        return True
    expected_hash = str(getattr(s, "access_token_sha256", None) or "").strip().lower()
    if not expected_hash:
        return False
    digest = hashlib.sha256(raw.encode("utf-8", "ignore")).hexdigest().lower()
    return secrets.compare_digest(digest, expected_hash)


def access_token_from_request(request: Request) -> str:
    header = str(request.headers.get("authorization") or "").strip()
    if header.lower().startswith("bearer "):
        token = header[7:].strip()
        if token:
            return token
    token = str(request.headers.get(AUTH_HEADER_NAME) or "").strip()
    if token:
        return token
    token = str(request.cookies.get(AUTH_COOKIE_NAME) or "").strip()
    if token:
        return token
    return str(request.query_params.get("access_token") or "").strip()


def request_is_authenticated(request: Request, settings: Settings | None = None) -> bool:
    return access_token_valid(access_token_from_request(request), settings=settings)


def auth_status_payload(request: Request | None = None, settings: Settings | None = None) -> dict:
    s = settings or auth_settings()
    authenticated = request_is_authenticated(request, settings=s) if request is not None else False
    return {
        "required": bool(getattr(s, "auth_required", False)),
        "configured": auth_token_configured(s),
        "authenticated": authenticated,
        "env": str(getattr(s, "app_env", "development") or "development"),
        "production": bool(getattr(s, "production", False)),
    }


def is_public_api_path(path: str) -> bool:
    clean = str(path or "")
    return clean in {
        "/api/health",
        "/api/auth/status",
        "/api/auth/login",
        "/api/auth/logout",
    }
