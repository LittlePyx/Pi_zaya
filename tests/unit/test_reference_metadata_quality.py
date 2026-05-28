from __future__ import annotations

import json

from api import reference_metadata_quality as mq


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
