from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import chat, generate, library, references, settings

app = FastAPI(title="Pi-zaya API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(generate.router)
app.include_router(library.router)
app.include_router(references.router)
app.include_router(settings.router)
