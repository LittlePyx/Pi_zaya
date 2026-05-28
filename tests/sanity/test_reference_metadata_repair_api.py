from __future__ import annotations

import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from api import reference_metadata_quality as mq
from api.main import app
from api.routers import library as library_router
from api.routers import references as references_router


class _ImmediateThread:
    def __init__(self, target, args=(), kwargs=None, **_kwargs):
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}

    def start(self) -> None:
        self.target(*self.args, **self.kwargs)


def _reset_backfill_state() -> None:
    with references_router._SHELF_METADATA_BACKFILL_LOCK:
        references_router._SHELF_METADATA_BACKFILL_STATE.clear()
        references_router._SHELF_METADATA_BACKFILL_STATE.update(
            {
                "ok": True,
                "status": "idle",
                "phase": "idle",
                "running": False,
                "progress": {"percent": 0, "processed": 0, "total": 0},
                "updated_at": 0.0,
            }
        )


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
    monkeypatch.setattr(library_router, "_quality_repair_runs_path", lambda: tmp_path / "repair_runs.jsonl")

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
    assert payload["impact"]["ready_delta"] == 1
    assert payload["acceptance"]["export_ready_after"] == 1
    assert payload["verification"]["type"] == "shelf_metadata_repair"
    assert payload["verification"]["quality_ok"] is True
    assert payload["repair_run"]["verification"]["export_ready_after"] == 1
    assert payload["impact"]["changed_fields"]


def test_shelf_metadata_repair_hydrates_from_reference_index(tmp_path, monkeypatch):
    def fake_enrich(detail):
        return dict(detail)

    db_dir = tmp_path / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    source_path = str(tmp_path / "md" / "source.en.md")
    (db_dir / "references_index.json").write_text(
        """
{
  "docs": {
    "source": {
      "path": "__SOURCE_PATH__",
      "name": "source.en.md",
      "refs": {
        "7": {
          "num": 7,
          "raw": "[7] Boyd S, Parikh N, Chu E. Alternating direction method of multipliers. Foundations and Trends in Machine Learning, 2011. doi:10.1561/2200000016",
          "title": "Alternating direction method of multipliers",
          "authors": "Boyd S, Parikh N, Chu E",
          "venue": "Foundations and Trends in Machine Learning",
          "year": "2011"
        }
      }
    }
  }
}
""".replace("__SOURCE_PATH__", source_path.replace("\\", "\\\\")),
        encoding="utf-8",
    )

    monkeypatch.setattr(mq, "enrich_citation_detail_meta", fake_enrich)
    monkeypatch.setattr(references_router, "get_settings", lambda: SimpleNamespace(db_dir=db_dir))
    monkeypatch.setattr(library_router, "_quality_repair_runs_path", lambda: tmp_path / "repair_runs.jsonl")

    client = TestClient(app)
    response = client.post(
        "/api/references/shelf/metadata/repair",
        json={
            "items": [
                {
                    "key": "source-r7",
                    "anchor": "ref-7",
                    "source_path": source_path,
                    "source_name": "source.en.md",
                    "ref_num": 7,
                    "title": "ADMM",
                }
            ]
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ready"] == 1
    assert payload["export_ready"] == 1
    assert payload["changed"] == 1
    assert payload["repair_run"]["phase"] == "shelf_metadata_verified"
    assert payload["impact"]["repair_sources"] == [{"name": "reference_index", "count": 1}]
    item = payload["items"][0]
    assert item["repair_sources"] == ["reference_index"]
    assert "doi" in item["changed_fields"]
    assert "missing_doi" in item["fixed_issue_codes"]
    assert item["after"]["status"] == "ready"
    assert item["meta"]["title"] == "Alternating direction method of multipliers"
    assert item["meta"]["doi"] == "10.1561/2200000016"
    assert item["meta"]["metadata_quality"]["status"] == "ready"


def test_bibliometrics_route_reuses_persisted_repair_cache(tmp_path, monkeypatch):
    db_dir = tmp_path / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    (db_dir / "crossref_cache.json").write_text(
        json.dumps(
            {
                "version": 1,
                "doi": {
                    "10.1561/2200000016": {
                        "title": "Alternating direction method of multipliers",
                        "authors": "Boyd S, Parikh N, Chu E",
                        "venue": "Foundations and Trends in Machine Learning",
                        "year": "2011",
                        "doi": "10.1561/2200000016",
                        "doi_url": "https://doi.org/10.1561/2200000016",
                        "summary_line": "The paper introduces the ADMM optimization framework.",
                        "summary_source": "abstract",
                        "summary_provider": "crossref",
                    }
                },
                "bib": {},
                "title": {},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    def fail_enrich(detail):
        raise AssertionError("route should reuse persisted repair metadata")

    monkeypatch.setattr(references_router, "get_settings", lambda: SimpleNamespace(db_dir=db_dir))
    monkeypatch.setattr(references_router, "enrich_citation_detail_meta", fail_enrich)

    client = TestClient(app)
    response = client.post(
        "/api/references/bibliometrics",
        json={
            "meta": {
                "key": "admm-cache",
                "source_path": "source.md",
                "raw": "[1] Boyd et al. doi:10.1561/2200000016",
            }
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["title"] == "Alternating direction method of multipliers"
    assert payload["authors"] == "Boyd S, Parikh N, Chu E"
    assert payload["venue"] == "Foundations and Trends in Machine Learning"
    assert payload["metadata_quality"]["status"] == "ready"
    assert payload["metadata_export_acceptance"]["export_ready"] is True
    assert payload["summary_source"] == "abstract"


def test_bibliometrics_route_attaches_quality_contract_to_enriched_result(tmp_path, monkeypatch):
    db_dir = tmp_path / "db"
    db_dir.mkdir(parents=True, exist_ok=True)

    def fake_enrich(detail):
        return {
            **dict(detail),
            "title": "The missing cone problem and low-pass distortion in optical serial sectioning microscopy",
            "authors": "Macias-Garza F, Bovik A C, Diller K R",
            "venue": "IEEE Transactions on Acoustics, Speech, and Signal Processing",
            "year": "1988",
            "doi": "10.1109/TASSP.1988.1164940",
            "doi_url": "https://doi.org/10.1109/TASSP.1988.1164940",
        }

    monkeypatch.setattr(references_router, "get_settings", lambda: SimpleNamespace(db_dir=db_dir))
    monkeypatch.setattr(references_router, "enrich_citation_detail_meta", fake_enrich)

    client = TestClient(app)
    response = client.post(
        "/api/references/bibliometrics",
        json={
            "meta": {
                "key": "missing-cone",
                "source_path": "source.md",
                "raw": "[3] Macias-Garza F et al. The missing cone problem and low-pass distortion. 1988.",
                "title": "Upstream reference",
            }
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["bibliometrics_checked"] is True
    assert payload["metadata_quality"]["status"] == "ready"
    assert payload["metadata_quality"]["ok"] is True
    assert payload["metadata_export_acceptance"]["export_ready"] is True
    assert payload["metadata_repair_status"] == "ready"
    assert payload["doi"] == "10.1109/TASSP.1988.1164940"


def test_shelf_metadata_backfill_route_scans_and_repairs_reference_index(tmp_path, monkeypatch):
    db_dir = tmp_path / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    source_path = tmp_path / "md" / "source.en.md"
    (db_dir / "references_index.json").write_text(
        json.dumps(
            {
                "version": 1,
                "docs": {
                    "source": {
                        "path": str(source_path),
                        "name": source_path.name,
                        "refs": {
                            "1": {
                                "num": 1,
                                "raw": "[1] Boyd et al. Alternating direction method of multipliers. 2011. doi:10.1561/2200000016",
                                "title": "Reference 1",
                            }
                        },
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

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
    monkeypatch.setattr(references_router, "get_settings", lambda: SimpleNamespace(db_dir=db_dir))
    monkeypatch.setattr(library_router, "_quality_repair_runs_path", lambda: tmp_path / "repair_runs.jsonl")

    client = TestClient(app)
    scan_response = client.get("/api/references/shelf/metadata/backfill/scan?limit=20")
    assert scan_response.status_code == 200
    scan_payload = scan_response.json()
    assert scan_payload["needs_repair"] == 1
    assert scan_payload["target_count"] == 1

    response = client.post(
        "/api/references/shelf/metadata/backfill",
        json={"limit": 10, "scan_limit": 20},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["requested"] == 1
    assert payload["changed"] == 1
    assert payload["export_ready"] == 1
    assert payload["verification"]["quality_ok"] is True
    assert payload["repair_run"]["phase"] == "shelf_metadata_verified"
    assert payload["after_scan"]["needs_repair"] == 0


def test_shelf_metadata_backfill_start_runs_background_job_and_reports_status(tmp_path, monkeypatch):
    _reset_backfill_state()
    db_dir = tmp_path / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    source_path = tmp_path / "md" / "source.en.md"
    (db_dir / "references_index.json").write_text(
        json.dumps(
            {
                "version": 1,
                "docs": {
                    "source": {
                        "path": str(source_path),
                        "name": source_path.name,
                        "refs": {
                            "1": {
                                "num": 1,
                                "raw": "[1] Boyd et al. Alternating direction method of multipliers. 2011. doi:10.1561/2200000016",
                                "title": "Reference 1",
                            }
                        },
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

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

    monkeypatch.setattr(references_router.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(mq, "enrich_citation_detail_meta", fake_enrich)
    monkeypatch.setattr(references_router, "get_settings", lambda: SimpleNamespace(db_dir=db_dir))
    monkeypatch.setattr(library_router, "_quality_repair_runs_path", lambda: tmp_path / "repair_runs.jsonl")

    client = TestClient(app)
    response = client.post(
        "/api/references/shelf/metadata/backfill/start",
        json={"limit": 10, "scan_limit": 20},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["started"] is True
    state = payload["state"]
    assert state["status"] == "completed"
    assert state["running"] is False
    assert state["progress"]["percent"] == 100
    assert state["result"]["requested"] == 1
    assert state["after_scan"]["needs_repair"] == 0

    status_response = client.get("/api/references/shelf/metadata/backfill/status")
    assert status_response.status_code == 200
    status = status_response.json()
    assert status["job_id"] == state["job_id"]
    assert status["result"]["verification"]["quality_ok"] is True
