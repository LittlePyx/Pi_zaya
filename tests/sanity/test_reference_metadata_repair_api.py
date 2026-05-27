from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from api import reference_metadata_quality as mq
from api.main import app
from api.routers import references as references_router


def test_shelf_metadata_repair_route_returns_quality_contract(tmp_path, monkeypatch):
    def fake_enrich(detail):
        return {
            **dict(detail),
            "title": "Alternating direction method of multipliers",
            "authors": "Boyd S, Parikh N, Chu E",
            "venue": "Foundations and Trends in Machine Learning",
            "year": "2011",
            "doi": "10.1561/2200000016",
            "doi_url": "https://doi.org/10.1561/2200000016",
        }

    monkeypatch.setattr(mq, "enrich_citation_detail_meta", fake_enrich)
    monkeypatch.setattr(references_router, "get_settings", lambda: SimpleNamespace(db_dir=tmp_path / "db"))

    client = TestClient(app)
    response = client.post(
        "/api/references/shelf/metadata/repair",
        json={
            "items": [
                {
                    "key": "admm",
                    "anchor": "a1",
                    "source_path": "scinerf.md",
                    "title": "ADMM",
                    "raw": "[1] Boyd et al. Alternating direction method of multipliers. 2011.",
                }
            ]
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["requested"] == 1
    assert payload["ready"] == 1
    assert payload["changed"] == 1
    item = payload["items"][0]
    assert item["key"] == "admm"
    assert item["after"]["status"] == "ready"
    assert item["meta"]["doi"] == "10.1561/2200000016"
    assert item["persisted"] is True
