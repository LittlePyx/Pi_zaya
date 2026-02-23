#!/usr/bin/env python3
from __future__ import annotations

import uvicorn
from pathlib import Path

from api.main import app

# Serve frontend static files in production
_DIST = Path(__file__).parent / "web" / "dist"
if _DIST.is_dir():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
