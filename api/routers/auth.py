from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from api.security import (
    AUTH_COOKIE_NAME,
    access_token_valid,
    auth_settings,
    auth_status_payload,
    auth_token_configured,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginBody(BaseModel):
    token: str


@router.get("/status")
def auth_status(request: Request):
    return auth_status_payload(request)


@router.post("/login")
def login(body: LoginBody, response: Response):
    settings = auth_settings()
    if not getattr(settings, "auth_required", False):
        response.delete_cookie(AUTH_COOKIE_NAME, path="/")
        return {"ok": True, **auth_status_payload(settings=settings)}
    if not auth_token_configured(settings):
        raise HTTPException(503, "API access token is not configured")
    token = str(body.token or "").strip()
    if not access_token_valid(token, settings=settings):
        raise HTTPException(401, "Invalid access token")
    response.set_cookie(
        AUTH_COOKIE_NAME,
        token,
        httponly=True,
        secure=bool(getattr(settings, "auth_cookie_secure", False)),
        samesite="lax",
        max_age=60 * 60 * 24 * 60,
        path="/",
    )
    payload = auth_status_payload(settings=settings)
    payload["authenticated"] = True
    return {"ok": True, **payload}


@router.post("/logout")
def logout(response: Response):
    response.delete_cookie(AUTH_COOKIE_NAME, path="/")
    return {"ok": True}
