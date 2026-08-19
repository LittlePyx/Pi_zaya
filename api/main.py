from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.types import Receive, Scope, Send

from kb.app_paths import configure_release_environment
from kb.version import read_app_version

configure_release_environment()

from api.routers import app as app_router
from api.routers import auth, chat, evidence_matrices, generate, library, maintenance, references, research_briefs, research_gaps, settings, user_issues
from api.security import auth_settings, auth_token_configured, is_public_api_path, request_is_authenticated
from kb.config import load_settings
from kb.retriever_cache import warm_retriever_async
from kb.task_runtime import _bg_reconcile_persisted_jobs
from kb.user_issue_store import start_remote_outbox_worker


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _bg_reconcile_persisted_jobs()
    except Exception:
        pass
    try:
        start_remote_outbox_worker(user_issues._issue_db_path())
    except Exception:
        pass
    try:
        warm_retriever_async(load_settings().db_dir)
    except Exception:
        pass
    yield


app = FastAPI(title="Pi-zaya API", version=read_app_version(), lifespan=lifespan)

_CORS_EXPOSE_HEADERS = [
    "Content-Disposition",
    "Server-Timing",
    "X-KB-Refs-Counts",
    "X-KB-Refs-Mode",
    "X-KB-Management-Auth",
]
_BODY_GUARD_METHODS = {"POST", "PUT", "PATCH"}
_USER_ISSUE_BODY_GUARD_PATHS = {"/api/user-issues", "/api/user-issues/ingest"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except Exception:
        return int(default)


def _bounded_env_int(name: str, default: int, *, min_value: int, max_value: int) -> int:
    value = _env_int(name, default)
    return max(min_value, min(max_value, value))


def _content_type_is_json(raw: str | None) -> bool:
    media_type = str(raw or "").split(";", 1)[0].strip().lower()
    return media_type == "application/json" or media_type.endswith("+json")


def _user_issue_body_size_limit() -> int:
    return _bounded_env_int("KB_USER_ISSUES_MAX_BODY_BYTES", 65_536, min_value=1024, max_value=1_048_576)


def _api_json_body_size_limit() -> int:
    return _bounded_env_int("KB_API_JSON_MAX_BODY_BYTES", 1_048_576, min_value=65_536, max_value=20 * 1024 * 1024)


def _scope_header(scope: Scope, name: bytes) -> str:
    target = name.lower()
    for key, value in scope.get("headers") or []:
        if bytes(key).lower() == target:
            try:
                return bytes(value).decode("latin-1").strip()
            except Exception:
                return ""
    return ""


def _request_body_limit_for_scope(scope: Scope) -> tuple[int, str]:
    method = str(scope.get("method") or "").upper()
    path = str(scope.get("path") or "")
    if method not in _BODY_GUARD_METHODS:
        return 0, ""
    if path in _USER_ISSUE_BODY_GUARD_PATHS:
        max_bytes = _user_issue_body_size_limit()
        return max_bytes, f"user issue payload is too large; max {max_bytes} bytes"
    if path.startswith("/api") and _content_type_is_json(_scope_header(scope, b"content-type")):
        max_bytes = _api_json_body_size_limit()
        return max_bytes, f"JSON request body is too large; max {max_bytes} bytes"
    return 0, ""


class RequestBodySizeGuardMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        max_bytes, detail = _request_body_limit_for_scope(scope)
        if max_bytes <= 0:
            await self.app(scope, receive, send)
            return

        raw_length = _scope_header(scope, b"content-length")
        if raw_length:
            try:
                content_length = max(0, int(raw_length))
            except Exception:
                content_length = 0
            if content_length > max_bytes:
                await JSONResponse({"detail": detail}, status_code=413)(scope, receive, send)
                return

        total = 0
        replay: list[dict] = []
        while True:
            message = await receive()
            replay.append(message)
            if message.get("type") == "http.request":
                chunk = message.get("body", b"") or b""
                total += len(chunk)
                if total > max_bytes:
                    await JSONResponse({"detail": detail}, status_code=413)(scope, receive, send)
                    return
                if not message.get("more_body", False):
                    break
            else:
                break

        async def replay_receive():
            if replay:
                return replay.pop(0)
            return {"type": "http.request", "body": b"", "more_body": False}

        await self.app(scope, replay_receive, send)


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
    expose_headers=_CORS_EXPOSE_HEADERS,
)
app.add_middleware(RequestBodySizeGuardMiddleware)


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
app.include_router(evidence_matrices.router)
app.include_router(generate.router)
app.include_router(library.router)
app.include_router(maintenance.router)
app.include_router(references.router)
app.include_router(research_briefs.router)
app.include_router(research_gaps.router)
app.include_router(settings.router)
app.include_router(user_issues.router)
