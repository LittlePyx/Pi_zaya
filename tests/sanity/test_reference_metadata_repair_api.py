from __future__ import annotations

import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from api import reference_metadata_quality as mq
from api.main import app
from api.routers import library as library_router
from api.routers import references as references_router
from kb.library_store import LibraryStore


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


def test_reference_metadata_routes_reject_unbounded_payloads_before_work(monkeypatch):
    def fail_repair(*args, **kwargs):
        raise AssertionError("repair should not run for invalid request bodies")

    def fail_backfill(*args, **kwargs):
        raise AssertionError("backfill should not run for invalid request bodies")

    def fail_polish(*args, **kwargs):
        raise AssertionError("polish should not run for invalid request bodies")

    monkeypatch.setattr(references_router, "repair_citation_metadata_batch", fail_repair)
    monkeypatch.setattr(references_router, "backfill_reference_metadata", fail_backfill)
    monkeypatch.setattr(references_router, "polish_citation_card_detail", fail_polish)

    client = TestClient(app)
    huge_bibliometrics = client.post(
        "/api/references/bibliometrics",
        json={"meta": {"title": "x" * 95_000}},
    )
    too_many_repair_items = client.post(
        "/api/references/shelf/metadata/repair",
        json={"items": [{"key": f"ref-{idx}", "title": "Demo"} for idx in range(121)]},
    )
    huge_repair_item = client.post(
        "/api/references/shelf/metadata/repair",
        json={"items": [{"key": "huge", "title": "x" * 45_000}]},
    )
    invalid_backfill_limit = client.post(
        "/api/references/shelf/metadata/backfill",
        json={"limit": 10_000},
    )
    huge_polish_meta = client.post(
        "/api/references/citation-card-polish",
        json={"meta": {"title": "x" * 95_000}},
    )

    assert huge_bibliometrics.status_code == 422
    assert too_many_repair_items.status_code == 422
    assert huge_repair_item.status_code == 422
    assert invalid_backfill_limit.status_code == 422
    assert huge_polish_meta.status_code == 422


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


def test_bibliometrics_route_enriches_ready_metadata_when_summary_missing(tmp_path, monkeypatch):
    db_dir = tmp_path / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    (db_dir / "crossref_cache.json").write_text(
        json.dumps(
            {
                "version": 1,
                "doi": {
                    "10.1364/oe.458742": {
                        "title": "Fast hyperspectral single-pixel imaging via frequency-division multiplexed illumination",
                        "authors": "Jiang X, Li Z, Du G, et al",
                        "venue": "Optics Express",
                        "year": "2022",
                        "doi": "10.1364/oe.458742",
                        "doi_url": "https://doi.org/10.1364/oe.458742",
                    }
                },
                "bib": {},
                "title": {},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    enrich_calls = []

    def fake_enrich(detail):
        enrich_calls.append(dict(detail))
        return {
            **dict(detail),
            "summary_line": "A grounded article summary from the Crossref abstract.",
            "summary_source": "abstract",
            "summary_provider": "crossref",
            "summary_quality": {
                "ok": True,
                "status": "grounded",
                "source": "abstract",
                "provider": "crossref",
                "export_ready": True,
                "issues": [],
            },
        }

    monkeypatch.setattr(references_router, "get_settings", lambda: SimpleNamespace(db_dir=db_dir))
    monkeypatch.setattr(references_router, "enrich_citation_detail_meta", fake_enrich)

    client = TestClient(app)
    response = client.post(
        "/api/references/bibliometrics",
        json={
            "meta": {
                "key": "oe-summary-missing",
                "source_path": "source.md",
                "raw": "[184] Jiang et al. doi:10.1364/oe.458742",
            }
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert enrich_calls
    assert enrich_calls[0]["doi"] == "10.1364/oe.458742"
    assert payload["summary_source"] == "abstract"
    assert payload["summary_provider"] == "crossref"
    assert payload["metadata_export_acceptance"]["export_ready"] is True
    assert payload["metadata_export_acceptance"]["summary_export_ready"] is True
    cache = json.loads((db_dir / "crossref_cache.json").read_text(encoding="utf-8"))
    cached = cache["doi"]["10.1364/oe.458742"]
    assert cached["summary_line"] == "A grounded article summary from the Crossref abstract."
    assert cached["summary_source"] == "abstract"


def test_bibliometrics_route_prefers_local_markdown_abstract_for_shelf_summary(tmp_path, monkeypatch):
    db_dir = tmp_path / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    md_dir = tmp_path / "md_output"
    md_dir.mkdir(parents=True, exist_ok=True)
    md_path = md_dir / "paper.en.md"
    md_path.write_text(
        "\n".join(
            [
                "# Fast hyperspectral single-pixel imaging",
                "",
                "## Abstract",
                "This paper proposes a frequency-division multiplexed illumination method for fast hyperspectral single-pixel imaging. "
                "The method reconstructs spectral images from multiplexed measurements and validates improved acquisition speed.",
                "",
                "## References",
                "[1] Other work.",
            ]
        ),
        encoding="utf-8",
    )

    def fake_enrich(detail):
        return dict(detail)

    monkeypatch.setattr(references_router, "get_settings", lambda: SimpleNamespace(db_dir=db_dir))
    monkeypatch.setattr(references_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(references_router, "enrich_citation_detail_meta", fake_enrich)

    client = TestClient(app)
    response = client.post(
        "/api/references/bibliometrics",
        json={
            "meta": {
                "key": "local-summary",
                "source_path": str(md_path),
                "title": "Fast hyperspectral single-pixel imaging",
                "authors": "Jiang X, Li Z",
                "venue": "Optics Express",
                "year": "2022",
                "doi": "10.1364/oe.458742",
                "summary_line": "This citation supports the current answer.",
                "summary_source": "citation_context",
            }
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary_source"] == "abstract"
    assert payload["summary_provider"] == "local_markdown"
    assert payload["summary_quality"]["status"] == "grounded"
    assert "frequency-division multiplexed illumination" in payload["summary_line"]
    assert payload["metadata_export_acceptance"]["summary_export_ready"] is True


def test_bibliometrics_route_rejects_local_summary_outside_reference_asset_roots(tmp_path, monkeypatch):
    db_dir = tmp_path / "db"
    md_dir = tmp_path / "md_output"
    db_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)
    outside_md = tmp_path / "project-not-library.md"
    outside_md.write_text(
        "# Internal note\n\n## Abstract\n\nThis private project note must not be exposed as citation summary.",
        encoding="utf-8",
    )

    def fake_enrich(detail):
        return dict(detail)

    monkeypatch.setattr(references_router, "get_settings", lambda: SimpleNamespace(db_dir=db_dir))
    monkeypatch.setattr(references_router, "_md_dir", lambda: md_dir)
    monkeypatch.setattr(references_router, "enrich_citation_detail_meta", fake_enrich)

    client = TestClient(app)
    response = client.post(
        "/api/references/bibliometrics",
        json={
            "meta": {
                "key": "outside-local-summary",
                "source_path": str(outside_md),
                "title": "Internal note",
                "authors": "Local User",
                "venue": "Notebook",
                "year": "2026",
                "summary_line": "This citation supports the current answer.",
                "summary_source": "citation_context",
            }
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload.get("summary_provider") != "local_markdown"
    assert "private project note" not in json.dumps(payload, ensure_ascii=False)


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


def test_bibliometrics_route_reports_local_library_doi_match(tmp_path, monkeypatch):
    db_dir = tmp_path / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    library_db = tmp_path / "library.sqlite3"
    pdf_path = tmp_path / "papers" / "Fast hyperspectral single-pixel imaging.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.4\n")
    LibraryStore(library_db).upsert(
        "sha1-local-oe",
        pdf_path,
        citation_meta={
            "title": "Fast hyperspectral single-pixel imaging via frequency-division multiplexed illumination",
            "authors": "Jiang X, Li Z, Du G, et al",
            "venue": "Optics Express",
            "year": "2022",
            "doi": "10.1364/oe.458742",
            "doi_url": "https://doi.org/10.1364/oe.458742",
        },
    )

    def fake_enrich(detail):
        return dict(detail)

    monkeypatch.setattr(
        references_router,
        "get_settings",
        lambda: SimpleNamespace(db_dir=db_dir, library_db_path=library_db),
    )
    monkeypatch.setattr(references_router, "enrich_citation_detail_meta", fake_enrich)

    client = TestClient(app)
    response = client.post(
        "/api/references/bibliometrics",
        json={
            "meta": {
                "key": "oe-local",
                "anchor": "ref-184",
                "source_path": "current-paper.md",
                "title": "Fast hyperspectral single-pixel imaging via frequency-division multiplexed illumination",
                "authors": "Jiang X, Li Z, Du G, et al",
                "venue": "Optics Express",
                "year": "2022",
                "doi": "https://doi.org/10.1364/oe.458742",
            }
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["library_match_status"] == "in_library"
    assert payload["library_match_method"] == "doi"
    assert payload["library_match_path"] == str(pdf_path)
    assert payload["library_match"]["matched"] is True
    assert payload["library_match"]["confidence"] >= 0.99


def test_bibliometrics_route_promotes_exact_library_doi_before_enrichment(tmp_path, monkeypatch):
    db_dir = tmp_path / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    library_db = tmp_path / "library.sqlite3"
    pdf_path = tmp_path / "papers" / "Principles and prospects for single-pixel imaging.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.4\n")
    LibraryStore(library_db).upsert(
        "sha1-local-natphoton",
        pdf_path,
        citation_meta={
            "title": "Principles and prospects for single-pixel imaging",
            "authors": "Edgar M, Gibson G, Padgett M",
            "venue": "Nature Photonics",
            "year": "2019",
            "doi": "10.1038/s41566-018-0300-7",
            "doi_url": "https://doi.org/10.1038/s41566-018-0300-7",
        },
    )
    enrich_calls = []

    def fake_enrich(detail):
        enrich_calls.append(dict(detail))
        return {
            **dict(detail),
            "citation_count": 642,
            "citation_source": "Crossref",
            "journal_if": 32.9,
            "journal_quartile": "Q1",
            "summary_line": "A grounded summary for the correctly matched article.",
            "summary_source": "abstract",
            "summary_provider": "crossref",
            "summary_quality": {
                "ok": True,
                "status": "grounded",
                "score": 94,
                "source": "abstract",
                "provider": "crossref",
                "locale": "zh",
                "export_ready": True,
            },
            "summary_locale": "zh",
        }

    monkeypatch.setattr(
        references_router,
        "get_settings",
        lambda: SimpleNamespace(db_dir=db_dir, library_db_path=library_db),
    )
    monkeypatch.setattr(
        references_router,
        "_local_source_summary_meta",
        lambda _meta, *, target_locale="": {
            "summary_line": "A grounded local summary that must not skip metric refresh.",
            "summary_source": "abstract",
            "summary_provider": "local_markdown",
            "summary_locale": target_locale or "zh",
            "summary_quality": {
                "ok": True,
                "status": "grounded",
                "score": 94,
                "source": "abstract",
                "provider": "local_markdown",
                "locale": target_locale or "zh",
                "export_ready": True,
            },
        },
    )
    monkeypatch.setattr(references_router, "enrich_citation_detail_meta", fake_enrich)

    client = TestClient(app)
    response = client.post(
        "/api/references/bibliometrics",
        json={
            "meta": {
                "key": "wrong-doi-upstream",
                "source_path": "citing-paper.en.md",
                "title": "Principles and prospects for single-pixel imaging",
                "authors": "M. Edgar, G. Gibson, M. Padgett",
                "venue": "Nat. Photonics",
                "year": "2019",
                "doi": "10.1126/science.4071051",
                "doi_url": "https://doi.org/10.1126/science.4071051",
                "citation_count": 1,
                "journal_if": 1.1,
                "summary_line": "Summary for the wrong DOI.",
                "summary_source": "crossref",
            }
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert enrich_calls
    assert enrich_calls[0]["doi"] == "10.1038/s41566-018-0300-7"
    assert payload["doi"] == "10.1038/s41566-018-0300-7"
    assert payload["library_match_previous_doi"] == "10.1126/science.4071051"
    assert payload["citation_count"] == 642
    assert payload["journal_if"] == 32.9
    assert payload["journal_quartile"] == "Q1"
    assert payload["summary_source"] == "abstract"


def test_bibliometrics_route_does_not_treat_inpaper_source_as_upstream_library_match(tmp_path, monkeypatch):
    db_dir = tmp_path / "db"
    db_dir.mkdir(parents=True, exist_ok=True)
    library_db = tmp_path / "library.sqlite3"
    current_pdf = tmp_path / "papers" / "Current paper.pdf"
    current_pdf.parent.mkdir(parents=True, exist_ok=True)
    current_pdf.write_bytes(b"%PDF-1.4\n")
    LibraryStore(library_db).upsert(
        "sha1-current",
        current_pdf,
        citation_meta={
            "title": "Current paper already in the local library",
            "year": "2025",
            "doi": "10.0000/current-paper",
        },
    )

    def fake_enrich(detail):
        return dict(detail)

    monkeypatch.setattr(
        references_router,
        "get_settings",
        lambda: SimpleNamespace(db_dir=db_dir, library_db_path=library_db),
    )
    monkeypatch.setattr(references_router, "enrich_citation_detail_meta", fake_enrich)

    client = TestClient(app)
    response = client.post(
        "/api/references/bibliometrics",
        json={
            "meta": {
                "key": "system-b-upstream",
                "anchor": "ref-upstream",
                "is_inpaper": True,
                "source_path": str(current_pdf),
                "source_name": current_pdf.name,
                "title": "Different upstream method without a local PDF",
                "year": "2021",
            }
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["library_match"]["matched"] is False
    assert payload["library_match_status"] in {"not_in_library", "unknown"}
    assert payload.get("library_match_path", "") == ""


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
