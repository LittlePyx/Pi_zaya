from __future__ import annotations

import os

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from api.routers import app as app_router
from api.routers import auth, chat, generate, library, maintenance, references, settings, user_issues
from api.security import auth_settings, auth_token_configured, is_public_api_path, request_is_authenticated

app = FastAPI(title="Pi-zaya API")


def _split_env_list(raw: str) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()]


def _cors_config() -> dict:
    raw_origins = os.environ.get("KB_API_ALLOW_ORIGINS") or os.environ.get("KB_CORS_ALLOW_ORIGINS") or ""
    if raw_origins.strip() == "*":
        return {"allow_origins": ["*"], "allow_origin_regex": None}
    origins = _split_env_list(raw_origins) or [
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ]
    local_regex = os.environ.get("KB_API_ALLOW_ORIGIN_REGEX") or r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$"
    return {"allow_origins": origins, "allow_origin_regex": local_regex}


_cors = _cors_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors["allow_origins"],
    allow_origin_regex=_cors["allow_origin_regex"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def api_access_guard(request: Request, call_next):
    path = request.url.path
    if request.method == "OPTIONS" or not path.startswith("/api") or is_public_api_path(path):
        return await call_next(request)
    s = auth_settings()
    if not bool(getattr(s, "auth_required", False)):
        return await call_next(request)
    if not auth_token_configured(s):
        return JSONResponse(
            {"detail": "API access token is not configured"},
            status_code=503,
        )
    if request_is_authenticated(request, settings=s):
        return await call_next(request)
    return JSONResponse(
        {"detail": "Authentication required"},
        status_code=401,
        headers={"WWW-Authenticate": "Bearer"},
    )


app.include_router(auth.router)
app.include_router(app_router.router)
app.include_router(chat.router)
app.include_router(generate.router)
app.include_router(library.router)
app.include_router(maintenance.router)
app.include_router(references.router)
app.include_router(settings.router)
app.include_router(user_issues.router)
