from __future__ import annotations

import hashlib
import secrets

from fastapi import Request

from kb.config import Settings, load_settings

AUTH_COOKIE_NAME = "kb_access_token"
AUTH_HEADER_NAME = "X-KB-Access-Token"
MANAGEMENT_AUTH_HEADER_NAME = "X-KB-Management-Token"


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


def management_auth_required(settings: Settings | None = None) -> bool:
    s = settings or auth_settings()
    return bool(getattr(s, "management_auth_required", False))


def management_token_configured(settings: Settings | None = None) -> bool:
    s = settings or auth_settings()
    return bool(
        getattr(s, "management_access_token", None)
        or getattr(s, "management_access_token_sha256", None)
        or getattr(s, "access_token", None)
        or getattr(s, "access_token_sha256", None)
    )


def management_token_valid(token: str | None, settings: Settings | None = None) -> bool:
    raw = str(token or "").strip()
    if not raw:
        return False
    s = settings or auth_settings()
    candidates = (
        (
            str(getattr(s, "management_access_token", None) or "").strip(),
            str(getattr(s, "management_access_token_sha256", None) or "").strip().lower(),
        ),
        (
            str(getattr(s, "access_token", None) or "").strip(),
            str(getattr(s, "access_token_sha256", None) or "").strip().lower(),
        ),
    )
    digest = hashlib.sha256(raw.encode("utf-8", "ignore")).hexdigest().lower()
    return any(
        (plain and secrets.compare_digest(raw, plain))
        or (hashed and secrets.compare_digest(digest, hashed))
        for plain, hashed in candidates
    )


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
    return ""


def management_token_from_request(request: Request) -> str:
    token = str(request.headers.get(MANAGEMENT_AUTH_HEADER_NAME) or "").strip()
    if token:
        return token
    return access_token_from_request(request)


def request_is_authenticated(request: Request, settings: Settings | None = None) -> bool:
    return access_token_valid(access_token_from_request(request), settings=settings)


def request_has_management_access(request: Request, settings: Settings | None = None) -> bool:
    return management_token_valid(management_token_from_request(request), settings=settings)


def auth_status_payload(request: Request | None = None, settings: Settings | None = None) -> dict:
    s = settings or auth_settings()
    required = bool(getattr(s, "auth_required", False))
    authenticated = request_is_authenticated(request, settings=s) if request is not None else False
    management_required = management_auth_required(s)
    management_authenticated = request_has_management_access(request, settings=s) if request is not None else False
    return {
        "required": required,
        "configured": auth_token_configured(s) if required else False,
        "authenticated": authenticated if required else False,
        "management_required": management_required,
        "management_configured": management_token_configured(s) if management_required else False,
        "management_authenticated": management_authenticated if management_required else False,
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
        "/api/user-issues/ingest",
    }
