from __future__ import annotations

import json

from api import reference_metadata_quality as mq


def test_metadata_accepts_initial_surname_author() -> None:
    quality = mq.citation_metadata_quality(
        {
            "source_path": "paper.en.md",
            "title": "Quantized Fourier ptychography with binary images from SPAD cameras",
            "authors": "X Yang",
            "venue": "Photon. Res.",
            "year": "2021",
            "doi": "10.1364/PRJ.427699",
        }
    )

    assert quality["status"] == "ready"
    acceptance = mq.citation_metadata_export_acceptance({"metadata_quality": quality, **{
        "source_path": "paper.en.md",
        "title": "Quantized Fourier ptychography with binary images from SPAD cameras",
        "authors": "X Yang",
        "venue": "Photon. Res.",
        "year": "2021",
        "doi": "10.1364/PRJ.427699",
    }})
    assert acceptance["field_ready"]["authors"] is True
    assert acceptance["export_ready"] is True


def test_metadata_accepts_single_word_journal_venue() -> None:
    detail = {
        "source_path": "spd_review.en.md",
        "title": (
            "High-performance waveguide coupled Germanium-on-silicon single-photon avalanche diode "
            "with independently controllable absorption and multiplication"
        ),
        "authors": "H Wang",
        "venue": "Nanophotonics",
        "year": "2023",
        "volume": "12",
        "issue": "4",
        "pages": "705",
        "doi": "10.1515/nanoph-2022-0663",
        "raw": (
            "H. Wang, Y. Shi, Y. Zuo, Y. Yu, L. Lei, X. Zhang, and Z. Qian, "
            "High-performance waveguide coupled Germanium-on-silicon single-photon avalanche diode "
            "with independently controllable absorption and multiplication, "
            "Nanophotonics 12(4), 705 (2023)."
        ),
        "cite_fmt": (
            "H Wang. High-performance waveguide coupled Germanium-on-silicon single-photon avalanche diode "
            "with independently controllable absorption and multiplication. Nanophotonics, 12(4):705 (2023)."
        ),
    }

    quality = mq.citation_metadata_quality(detail)
    acceptance = mq.citation_metadata_export_acceptance({"metadata_quality": quality, **detail})

    assert "venue" not in quality["missing_fields"]
    assert acceptance["field_ready"]["venue"] is True
    assert "venue" not in acceptance["missing_fields"]
    assert acceptance["export_ready"] is True


def test_metadata_parses_authors_from_raw_reference_prefix() -> None:
    raw = (
        "Benjamin Sussman and Erik M. Gauger. Pattern Analysis and Plenoptic Imaging: "
        "a mini review of light-field and ultra-slim light field microscopy that combines "
        "computation and light-field illumination. Wave Optics, Oct 2013."
    )

    quality = mq.citation_metadata_quality(
        {
            "source_path": "qclfm.en.md",
            "title": (
                "Pattern Analysis and Plenoptic Imaging: a mini review of light-field and "
                "ultra-slim light field microscopy that combines computation and light-field illumination"
            ),
            "venue": "Wave Optics, Oct",
            "year": "2013",
            "raw": raw,
        }
    )

    assert "authors" not in quality["missing_fields"]
    acceptance = mq.citation_metadata_export_acceptance(
        {
            "source_path": "qclfm.en.md",
            "title": (
                "Pattern Analysis and Plenoptic Imaging: a mini review of light-field and "
                "ultra-slim light field microscopy that combines computation and light-field illumination"
            ),
            "venue": "Wave Optics, Oct",
            "year": "2013",
            "raw": raw,
            "metadata_quality": quality,
        }
    )
    assert acceptance["field_ready"]["authors"] is True
    assert acceptance["field_ready"]["doi"] is False


def test_metadata_does_not_treat_bare_et_al_prefix_as_export_ready_authors() -> None:
    detail = {
        "source_path": "demo.en.md",
        "title": "Single-shot compressive spectral imaging",
        "raw": "Gehm et al. Single-shot compressive spectral imaging.",
    }

    quality = mq.citation_metadata_quality(detail)
    acceptance = mq.citation_metadata_export_acceptance({"metadata_quality": quality, **detail})

    assert "authors" in quality["missing_fields"]
    assert acceptance["field_ready"]["authors"] is False


def test_repair_promotes_doi_from_reference_text(monkeypatch):
    raw = (
        "[24] Gehm M, Brady D. Single-shot compressive spectral imaging with a "
        "dual-disperser architecture. Optics Express, 2007. doi:10.1364/OE.15.014013"
    )

    def fake_enrich(detail):
        assert "10.1364/OE.15.014013" in str(detail.get("raw") or "")
        return {
            **dict(detail),
            "title": "Single-shot compressive spectral imaging with a dual-disperser architecture",
            "authors": "Gehm M, Brady D",
            "venue": "Optics Express",
            "year": "2007",
            "doi": "10.1364/OE.15.014013",
            "doi_url": "https://doi.org/10.1364/OE.15.014013",
        }

    monkeypatch.setattr(mq, "enrich_citation_detail_meta", fake_enrich)

    result = mq.repair_citation_metadata_item(
        {
            "key": "demo",
            "anchor": "a1",
            "source_path": "paper.md",
            "title": "Reference 24",
            "raw": raw,
        }
    )

    assert result["ok"] is True
    assert result["repair_status"] == "repaired"
    assert "doi" in result["changed_fields"]
    assert result["before"]["status"] == "error"
    assert result["after"]["status"] == "ready"
    assert result["meta"]["doi"] == "10.1364/OE.15.014013"


def test_repair_promotes_doi_from_card_reference_entry(monkeypatch):
    entry = (
        "[24] M. E. Gehm and D. J. Brady. Single-shot compressive spectral imaging "
        "with a dual-disperser architecture. Optics Express, 2007. doi:10.1364/OE.15.014013"
    )
    monkeypatch.setattr(mq, "enrich_citation_detail_meta", lambda detail: dict(detail))

    result = mq.repair_citation_metadata_item(
        {
            "key": "demo-card-entry",
            "anchor": "a24",
            "source_path": "paper.md",
            "title": "Single-shot compressive spectral imaging with a dual-disperser architecture",
            "authors": "M. E. Gehm and D. J. Brady",
            "venue": "Optics Express",
            "year": "2007",
            "card_reference_entry": entry,
        }
    )

    assert result["ok"] is True
    assert result["repair_status"] == "repaired"
    assert "doi" in result["changed_fields"]
    assert result["meta"]["doi"] == "10.1364/OE.15.014013"
    assert result["meta"]["doi_url"] == "https://doi.org/10.1364/OE.15.014013"


def test_repair_classifies_connection_errors_as_retryable(monkeypatch):
    def fake_enrich(detail):
        raise ConnectionError("Crossref connection refused")

    monkeypatch.setattr(mq, "enrich_citation_detail_meta", fake_enrich)

    result = mq.repair_citation_metadata_item(
        {
            "key": "demo",
            "anchor": "a1",
            "source_path": "paper.md",
            "title": "Single-shot compressive spectral imaging",
            "raw": "[1] Demo reference, 2007.",
        }
    )

    assert result["ok"] is False
    assert result["repair_status"] == "retryable"
    assert result["retryable"] is True
    assert result["error_kind"] == "connection"


def test_repair_persists_reference_index_and_crossref_cache(tmp_path, monkeypatch):
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    source_path = tmp_path / "paper.en.md"
    raw = (
        "[24] Gehm M, Brady D. Single-shot compressive spectral imaging with a "
        "dual-disperser architecture. Optics Express, 2007. doi:10.1364/OE.15.014013"
    )
    (db_dir / "references_index.json").write_text(
        json.dumps(
            {
                "version": 1,
                "docs": {
                    str(source_path).lower(): {
                        "path": str(source_path),
                        "name": source_path.name,
                        "refs": {
                            "24": {
                                "num": 24,
                                "raw": raw,
                                "title": "Reference 24",
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
            "title": "Single-shot compressive spectral imaging with a dual-disperser architecture",
            "authors": "Gehm M, Brady D",
            "venue": "Optics Express",
            "year": "2007",
            "doi": "10.1364/OE.15.014013",
            "doi_url": "https://doi.org/10.1364/OE.15.014013",
        }

    monkeypatch.setattr(mq, "enrich_citation_detail_meta", fake_enrich)

    result = mq.repair_citation_metadata_batch(
        [
            {
                "key": "demo",
                "source_path": str(source_path),
                "num": 24,
                "title": "Reference 24",
                "raw": raw,
            }
        ],
        db_dir=db_dir,
    )

    assert result["persisted"] == 1
    item = result["items"][0]
    assert sorted(item["persisted_targets"]) == ["crossref_cache", "reference_index"]
    index_data = json.loads((db_dir / "references_index.json").read_text(encoding="utf-8"))
    ref = next(iter(index_data["docs"].values()))["refs"]["24"]
    assert ref["doi"] == "10.1364/OE.15.014013"
    assert ref["authors"] == "Gehm M, Brady D"
    assert ref["venue"] == "Optics Express"
    assert ref["crossref_ok"] is True
    cache = json.loads((db_dir / "crossref_cache.json").read_text(encoding="utf-8"))
    cached = cache["doi"]["10.1364/oe.15.014013"]
    assert cached["title"] == "Single-shot compressive spectral imaging with a dual-disperser architecture"
    assert cached["venue"] == "Optics Express"


def test_repair_hydrates_from_crossref_cache_without_network(tmp_path, monkeypatch):
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    (db_dir / "crossref_cache.json").write_text(
        json.dumps(
            {
                "version": 1,
                "doi": {
                    "10.1364/oe.15.014013": {
                        "title": "Single-shot compressive spectral imaging with a dual-disperser architecture",
                        "authors": "Gehm M, Brady D",
                        "venue": "Optics Express",
                        "year": "2007",
                        "doi": "10.1364/OE.15.014013",
                        "doi_url": "https://doi.org/10.1364/OE.15.014013",
                    }
                },
                "bib": {},
                "title": {},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mq, "enrich_citation_detail_meta", lambda detail: dict(detail))

    result = mq.repair_citation_metadata_item(
        {
            "key": "cache-demo",
            "anchor": "a24",
            "source_path": "paper.md",
            "title": "Reference 24",
            "raw": "[24] Gehm M, Brady D. doi:10.1364/OE.15.014013",
        },
        db_dir=db_dir,
    )

    assert result["ok"] is True
    assert "crossref_cache:doi" in result["repair_sources"]
    assert result["meta"]["authors"] == "Gehm M, Brady D"
    assert result["meta"]["venue"] == "Optics Express"
    assert result["meta"]["metadata_quality"]["status"] == "ready"


def test_repair_reuses_ready_persisted_metadata_without_enrichment(tmp_path, monkeypatch):
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    (db_dir / "crossref_cache.json").write_text(
        json.dumps(
            {
                "version": 1,
                "doi": {
                    "10.1364/oe.15.014013": {
                        "title": "Single-shot compressive spectral imaging with a dual-disperser architecture",
                        "authors": "Gehm M, Brady D",
                        "venue": "Optics Express",
                        "year": "2007",
                        "doi": "10.1364/OE.15.014013",
                        "doi_url": "https://doi.org/10.1364/OE.15.014013",
                        "summary_line": "The paper introduces a single-shot compressive spectral imaging design.",
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
        raise AssertionError("persisted metadata should be enough")

    monkeypatch.setattr(mq, "enrich_citation_detail_meta", fail_enrich)

    result = mq.repair_citation_metadata_item(
        {
            "key": "cache-ready",
            "source_path": "paper.md",
            "raw": "[24] Gehm M, Brady D. doi:10.1364/OE.15.014013",
        },
        db_dir=db_dir,
    )

    assert result["ok"] is True
    assert result["repair_status"] == "repaired"
    assert "crossref_cache:doi" in result["repair_sources"]
    assert result["meta"]["metadata_quality"]["status"] == "ready"
    assert result["meta"]["metadata_export_acceptance"]["export_ready"] is True
    assert result["meta"]["summary_source"] == "abstract"


def test_repair_batch_reports_export_acceptance(tmp_path, monkeypatch):
    monkeypatch.setattr(mq, "enrich_citation_detail_meta", lambda detail: {
        **dict(detail),
        "title": "Single-shot compressive spectral imaging with a dual-disperser architecture",
        "authors": "Gehm M, Brady D",
        "venue": "Optics Express",
        "year": "2007",
        "doi": "10.1364/OE.15.014013",
        "doi_url": "https://doi.org/10.1364/OE.15.014013",
        "summary_line": "The abstract describes a single-shot compressive spectral imaging architecture.",
        "summary_source": "abstract",
        "summary_quality": {
            "contract_version": 1,
            "ok": True,
            "status": "grounded",
            "score": 94,
            "source": "abstract",
            "issues": [],
            "export_ready": True,
        },
    })

    result = mq.repair_citation_metadata_batch(
        [
            {
                "key": "demo",
                "source_path": "paper.md",
                "title": "Reference 24",
                "raw": "[24] Gehm M, Brady D. Optics Express, 2007. doi:10.1364/OE.15.014013",
            }
        ],
        db_dir=tmp_path / "db",
    )

    assert result["export_ready"] == 1
    assert result["acceptance"]["quality_ok"] is True
    assert result["acceptance"]["export_ready_after"] == 1
    assert result["acceptance"]["summary_export_ready_after"] == 1
    assert result["impact"]["export_ready_delta"] == 1
    assert result["items"][0]["export_acceptance"]["missing_fields"] == []


def test_repair_normalizes_crossref_style_cache_record(tmp_path, monkeypatch):
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    (db_dir / "crossref_cache.json").write_text(
        json.dumps(
            {
                "version": 1,
                "doi": {
                    "10.1364/oe.15.014013": {
                        "DOI": "10.1364/OE.15.014013",
                        "URL": "https://doi.org/10.1364/OE.15.014013",
                        "title": [
                            "Single-shot compressive spectral imaging with a dual-disperser architecture"
                        ],
                        "author": [
                            {"family": "Gehm", "given": "M. E."},
                            {"family": "Brady", "given": "D. J."},
                        ],
                        "container-title": ["Optics Express"],
                        "published-print": {"date-parts": [[2007, 10, 29]]},
                        "volume": "15",
                        "issue": "22",
                        "page": "14013-14027",
                        "is-referenced-by-count": 245,
                    }
                },
                "bib": {},
                "title": {},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mq, "enrich_citation_detail_meta", lambda detail: dict(detail))

    result = mq.repair_citation_metadata_item(
        {
            "key": "crossref-style",
            "anchor": "a24",
            "source_path": "paper.md",
            "title": "Reference 24",
            "raw": "[24] Gehm M, Brady D. doi:10.1364/OE.15.014013",
        },
        db_dir=db_dir,
    )

    assert result["ok"] is True
    assert "crossref_cache:doi" in result["repair_sources"]
    assert result["after"]["status"] == "ready"
    assert result["meta"]["doi"] == "10.1364/OE.15.014013"
    assert result["meta"]["doi_url"] == "https://doi.org/10.1364/OE.15.014013"
    assert result["meta"]["title"] == "Single-shot compressive spectral imaging with a dual-disperser architecture"
    assert result["meta"]["authors"] == "Gehm M E, Brady D J"
    assert result["meta"]["venue"] == "Optics Express"
    assert result["meta"]["year"] == "2007"
    assert result["meta"]["pages"] == "14013-14027"
    assert result["meta"]["citation_count"] == 245


def test_repair_normalizes_crossref_source_reference_index_record(tmp_path, monkeypatch):
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    source_path = tmp_path / "source.en.md"
    (db_dir / "references_index.json").write_text(
        json.dumps(
            {
                "version": 1,
                "docs": {
                    "source": {
                        "path": str(source_path),
                        "name": source_path.name,
                        "refs": {
                            "24": {
                                "num": 24,
                                "DOI": "10.1364/OE.15.014013",
                                "article-title": "Single-shot compressive spectral imaging with a dual-disperser architecture",
                                "author": "M. E. Gehm and D. J. Brady",
                                "journal-title": "Optics Express",
                                "year": "2007",
                                "volume": "15",
                                "first-page": "14013",
                                "raw": "[24] Gehm and Brady. doi:10.1364/OE.15.014013",
                            }
                        },
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mq, "enrich_citation_detail_meta", lambda detail: dict(detail))

    result = mq.repair_citation_metadata_item(
        {
            "key": "source-r24",
            "anchor": "a24",
            "source_path": str(source_path),
            "ref_num": 24,
            "title": "Reference 24",
        },
        db_dir=db_dir,
    )

    assert result["ok"] is True
    assert result["repair_sources"] == ["reference_index"]
    assert result["meta"]["doi"] == "10.1364/OE.15.014013"
    assert result["meta"]["authors"] == "M. E. Gehm and D. J. Brady"
    assert result["meta"]["venue"] == "Optics Express"
    assert result["meta"]["year"] == "2007"
    assert result["meta"]["pages"] == "14013"


def test_scan_reference_metadata_backfill_targets_reads_full_reference_index(tmp_path, monkeypatch):
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    source_path = tmp_path / "paper.en.md"
    (db_dir / "references_index.json").write_text(
        json.dumps(
            {
                "version": 1,
                "docs": {
                    "paper": {
                        "path": str(source_path),
                        "name": source_path.name,
                        "refs": {
                            "1": {
                                "num": 1,
                                "raw": "[1] Boyd et al. Alternating direction method of multipliers. 2011. doi:10.1561/2200000016",
                                "title": "Reference 1",
                            },
                            "2": {
                                "num": 2,
                                "raw": "[2] Gehm M, Brady D. Single-shot compressive spectral imaging. Optics Express, 2007. doi:10.1364/OE.15.014013",
                                "title": "Single-shot compressive spectral imaging",
                                "authors": "Gehm M, Brady D",
                                "venue": "Optics Express",
                                "year": "2007",
                                "doi": "10.1364/OE.15.014013",
                            },
                        },
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mq, "enrich_citation_detail_meta", lambda detail: dict(detail))

    scan = mq.scan_reference_metadata_backfill_targets(db_dir=db_dir, limit=20)

    assert scan["scanned"] == 2
    assert scan["export_ready"] == 1
    assert scan["needs_repair"] == 1
    assert scan["target_count"] == 1
    assert scan["targets"][0]["ref_num"] == "1"
    assert scan["targets"][0]["source_path"] == str(source_path)
    assert {item["name"] for item in scan["missing_fields"]} >= {"authors", "venue", "year"}


def test_scan_reference_metadata_backfill_targets_excludes_expected_no_doi_sources(tmp_path, monkeypatch):
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    source_path = tmp_path / "paper.en.md"
    (db_dir / "references_index.json").write_text(
        json.dumps(
            {
                "version": 1,
                "docs": {
                    "paper": {
                        "path": str(source_path),
                        "name": source_path.name,
                        "refs": {
                            "1": {
                                "num": 1,
                                "raw": (
                                    "[1] OpenAI. GPT-4 technical report. Technical report, 2023. "
                                    "https://openai.com/research/gpt-4"
                                ),
                                "title": "GPT-4 technical report",
                                "authors": "OpenAI",
                                "venue": "OpenAI",
                                "year": "2023",
                                "metadata_status": "non_article_source_ok",
                                "missing_reason": "no_doi_expected",
                                "metadata_action": "non_article_ok",
                            }
                        },
                    }
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mq, "enrich_citation_detail_meta", lambda detail: dict(detail))

    scan = mq.scan_reference_metadata_backfill_targets(db_dir=db_dir, limit=20)

    assert scan["scanned"] == 1
    assert scan["needs_repair"] == 0
    assert scan["target_count"] == 0
    assert scan["non_article_ok"] == 1
    assert {item["name"] for item in scan["missing_reasons"]} == {"no_doi_expected"}


def test_backfill_reference_metadata_updates_reference_index_from_scan(tmp_path, monkeypatch):
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    source_path = tmp_path / "paper.en.md"
    (db_dir / "references_index.json").write_text(
        json.dumps(
            {
                "version": 1,
                "docs": {
                    "paper": {
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

    result = mq.backfill_reference_metadata(db_dir=db_dir, limit=10, scan_limit=20)

    assert result["requested"] == 1
    assert result["changed"] == 1
    assert result["export_ready"] == 1
    assert result["scan"]["needs_repair"] == 1
    assert result["after_scan"]["needs_repair"] == 0
    index_data = json.loads((db_dir / "references_index.json").read_text(encoding="utf-8"))
    ref = index_data["docs"]["paper"]["refs"]["1"]
    assert ref["title"] == "Alternating direction method of multipliers"
    assert ref["authors"] == "Boyd S, Parikh N, Chu E"
    assert ref["venue"] == "Foundations and Trends in Machine Learning"
    assert ref["doi"] == "10.1561/2200000016"


def test_backfill_reference_metadata_persists_ready_crossref_cache_to_index(tmp_path, monkeypatch):
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    source_path = tmp_path / "paper.en.md"
    (db_dir / "references_index.json").write_text(
        json.dumps(
            {
                "version": 1,
                "docs": {
                    "paper": {
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
        raise AssertionError("ready cache should be persisted without network enrichment")

    monkeypatch.setattr(mq, "enrich_citation_detail_meta", fail_enrich)

    result = mq.backfill_reference_metadata(db_dir=db_dir, limit=10, scan_limit=20)

    assert result["requested"] == 1
    assert result["persisted"] == 1
    assert result["preheated"] == 1
    assert result["after_scan"]["needs_repair"] == 0
    item = result["items"][0]
    assert "crossref_cache:doi" in item["repair_sources"]
    assert "reference_index" in item["persisted_targets"]
    index_data = json.loads((db_dir / "references_index.json").read_text(encoding="utf-8"))
    ref = index_data["docs"]["paper"]["refs"]["1"]
    assert ref["authors"] == "Boyd S, Parikh N, Chu E"
    assert ref["venue"] == "Foundations and Trends in Machine Learning"
    assert ref["doi"] == "10.1561/2200000016"


def test_metadata_quality_hides_candidate_external_status_when_visible_identity_is_complete():
    quality = mq.citation_metadata_quality(
        {
            "source_path": "paper.md",
            "title": "Single-shot compressive spectral imaging with a dual-disperser architecture",
            "authors": "Gehm M, Brady D",
            "venue": "Optics Express",
            "year": "2007",
            "doi": "10.1364/OE.15.014013",
            "external_metadata_status": "candidate",
            "external_metadata_reason": "Bibliographic search was below the trust threshold.",
            "external_doi": "10.1364/OE.15.014013",
        }
    )

    assert quality["status"] == "ready"
    assert all(
        str(issue.get("code") or "") != "external_metadata_candidate"
        for issue in quality["issues"]
    )


def test_metadata_quality_keeps_review_when_candidate_doi_conflicts():
    quality = mq.citation_metadata_quality(
        {
            "source_path": "paper.md",
            "title": "Single-shot compressive spectral imaging with a dual-disperser architecture",
            "authors": "Gehm M, Brady D",
            "venue": "Optics Express",
            "year": "2007",
            "doi": "10.1364/OE.15.014013",
            "external_metadata_status": "candidate",
            "external_doi": "10.1000/wrong",
        }
    )

    assert quality["status"] == "warning"
    assert any(
        str(issue.get("code") or "") == "external_metadata_candidate"
        for issue in quality["issues"]
    )
