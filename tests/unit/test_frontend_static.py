from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.frontend_static import FrontendStaticFiles


@pytest.mark.parametrize("path", ["/", "/library", "/notes", "/reader/session/abc-123"])
def test_browser_deep_links_serve_react_entry(tmp_path, path):
    (tmp_path / "index.html").write_text("<html>React entry</html>", encoding="utf-8")
    app = FastAPI()
    app.mount("/", FrontendStaticFiles(directory=tmp_path, html=True))
    with TestClient(app) as client:
        response = client.get(path)
    assert response.status_code == 200
    assert "React entry" in response.text


@pytest.mark.parametrize("path", ["/api/missing", "/assets/missing.js", "/missing", "/reader/session/abc/missing"])
def test_missing_api_assets_and_unknown_routes_keep_404(tmp_path, path):
    (tmp_path / "index.html").write_text("<html>React entry</html>", encoding="utf-8")
    app = FastAPI()
    app.mount("/", FrontendStaticFiles(directory=tmp_path, html=True))
    with TestClient(app) as client:
        assert client.get(path).status_code == 404
