from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

from api.security import (
    AUTH_COOKIE_NAME,
    access_token_valid,
    auth_settings,
    auth_status_payload,
    auth_token_configured,
    management_auth_required,
    management_token_configured,
    management_token_valid,
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
    requires_user_auth = bool(getattr(settings, "auth_required", False))
    requires_management_auth = management_auth_required(settings)
    if not requires_user_auth and not requires_management_auth:
        response.delete_cookie(AUTH_COOKIE_NAME, path="/")
        return {"ok": True, **auth_status_payload(settings=settings)}
    if requires_user_auth and not auth_token_configured(settings):
        raise HTTPException(503, "API access token is not configured")
    if requires_management_auth and not management_token_configured(settings):
        raise HTTPException(503, "Management access token is not configured")
    token = str(body.token or "").strip()
    if not (
        access_token_valid(token, settings=settings)
        or management_token_valid(token, settings=settings)
    ):
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
    if requires_user_auth and access_token_valid(token, settings=settings):
        payload["authenticated"] = True
    if requires_management_auth and management_token_valid(token, settings=settings):
        payload["management_authenticated"] = True
    return {"ok": True, **payload}


@router.post("/logout")
def logout(response: Response):
    response.delete_cookie(AUTH_COOKIE_NAME, path="/")
    return {"ok": True}
