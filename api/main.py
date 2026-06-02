from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import chat, generate, library, references, settings

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

app.include_router(chat.router)
app.include_router(generate.router)
app.include_router(library.router)
app.include_router(references.router)
app.include_router(settings.router)
