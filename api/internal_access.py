from __future__ import annotations

import os

from fastapi import HTTPException, Request

from api.security import auth_settings, request_is_authenticated


def _env_bool(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


def require_internal_api(request: Request) -> None:
    """Gate maintenance/diagnostic APIs that should not be user-facing."""

    if internal_api_allowed(request):
        return

    settings = auth_settings()
    if bool(getattr(settings, "auth_required", False)):
        raise HTTPException(status_code=401, detail="Internal API authentication required")

    raise HTTPException(status_code=404, detail="Not found")


def internal_api_allowed(request: Request) -> bool:
    """Return whether this request may receive internal diagnostics."""

    settings = auth_settings()
    if bool(getattr(settings, "auth_required", False)):
        return bool(request_is_authenticated(request, settings=settings))

    if not bool(getattr(settings, "production", False)) and _env_bool("KB_ENABLE_INTERNAL_API", False):
        return True

    return False
