#!/usr/bin/env python3
from __future__ import annotations

import os
import uvicorn
from pathlib import Path

from api.main import app

# Serve frontend static files in production
_DIST = Path(__file__).parent / "web" / "dist"
if _DIST.is_dir():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="static")

if __name__ == "__main__":
    host = os.environ.get("KB_SERVER_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.environ.get("KB_SERVER_PORT", "8000"))
    reload_enabled = os.environ.get("KB_SERVER_RELOAD", "1").strip().lower() not in {"0", "false", "no", "off"}
    uvicorn.run("server:app", host=host, port=port, reload=reload_enabled)
