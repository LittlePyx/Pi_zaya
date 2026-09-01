from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json

from api import reference_metadata_quality as mq


def test_concurrent_summary_persistence_keeps_every_successful_abstract(tmp_path) -> None:
    db_dir = tmp_path / "db"
    db_dir.mkdir()

    def persist(index: int) -> None:
        mq.persist_repaired_citation_metadata(
            {
                "title": f"Concurrent abstract {index}",
                "doi": f"10.1000/concurrent-{index}",
                "summary_line": f"Verified abstract {index}.",
                "summary_source": "abstract",
                "summary_provider": "datacite",
                "summary_fetch_status": "ready",
                "summary_locale": "en",
            },
            db_dir=db_dir,
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(persist, range(16)))

    cache = json.loads((db_dir / "crossref_cache.json").read_text(encoding="utf-8"))
    assert len(cache["doi"]) == 16
    for index in range(16):
        record = cache["doi"][f"10.1000/concurrent-{index}"]
        assert record["summary_line"] == f"Verified abstract {index}."
        assert record["summary_fetch_status"] == "ready"
        assert record["summary_locale"] == "en"


def test_summary_acceptance_downgrades_stale_ready_contract_when_text_is_missing() -> None:
    detail = {
        "source_path": "paper.en.md",
        "title": "A complete article identity",
        "authors": "A Researcher",
        "venue": "Journal of Tests",
        "year": "2025",
        "doi": "10.1000/summary-missing",
        "summary_quality": {
            "ok": True,
            "status": "grounded",
            "source": "abstract",
            "provider": "crossref",
            "export_ready": True,
        },
    }

    acceptance = mq.citation_metadata_export_acceptance(detail)

    assert acceptance["summary_export_ready"] is False
    assert acceptance["summary_status"] == "missing"
    assert acceptance["summary"] == {
        "present": False,
        "export_ready": False,
        "status": "missing",
        "score": 0,
        "source": "",
        "provider": "",
        "issues": ["summary_missing"],
    }


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


def test_metadata_rejects_and_repairs_doi_conflicting_with_exact_library_match() -> None:
    detail = {
        "source_path": "citing-paper.en.md",
        "title": "Principles and prospects for single-pixel imaging",
        "authors": "M. Edgar, G. Gibson, M. Padgett",
        "venue": "Nat. Photonics",
        "year": "2019",
        "doi": "10.1126/science.4071051",
        "doi_url": "https://doi.org/10.1126/science.4071051",
        "citation_count": 7,
        "journal_if": 1.1,
        "summary_line": "Summary fetched for the wrong DOI.",
        "summary_source": "crossref",
        "library_match_status": "in_library",
        "library_match_confidence": 0.9,
        "library_match_method": "title_year",
        "library_match_reason": "title_exact",
        "library_match_title": "Principles and prospects for single-pixel imaging",
        "library_match_year": "2019",
        "library_match_doi": "10.1038/s41566-018-0300-7",
    }

    quality = mq.citation_metadata_quality(detail)
    acceptance = mq.citation_metadata_export_acceptance({**detail, "metadata_quality": quality})
    repaired = mq.promote_trusted_library_match_identity(detail)

    assert quality["status"] == "error"
    assert "library_match_doi_conflict" in {item["code"] for item in quality["issues"]}
    assert acceptance["field_ready"]["doi"] is False
    assert acceptance["export_ready"] is False
    assert repaired["doi"] == "10.1038/s41566-018-0300-7"
    assert repaired["doi_url"] == "https://doi.org/10.1038/s41566-018-0300-7"
    assert repaired["library_match_previous_doi"] == "10.1126/science.4071051"
    assert repaired["library_match_doi_promoted"] is True
    assert "citation_count" not in repaired
    assert "journal_if" not in repaired
    assert "summary_line" not in repaired


def test_metadata_rejects_related_work_doi_and_recovers_raw_reference_identity(monkeypatch) -> None:
    raw = (
        "Macias-Garza, F., Bovik, A. C., Diller, K. R., Aggarwal, S. J. & Aggarwal, J. K. "
        "The missing cone problem and low-pass distortion in optical serial sectioning microscopy. "
        "IEEE Trans. Acoust., Speech, Signal Process. 2, 890-893 (1988)."
    )
    detail = {
        "source_path": "citing-paper.en.md",
        "citation_route": "system_b",
        "is_inpaper": True,
        "raw": raw,
        "title": "Missing Cone Of Frequencies And Low-Pass Distortion In Three-Dimensional Microscopic Images",
        "authors": "Macias-Garza F, Diller K, Bovik A",
        "venue": "Optical Engineering",
        "year": "1988",
        "volume": "27",
        "pages": "890-893",
        "doi": "10.1117/12.7976703",
        "doi_url": "https://doi.org/10.1117/12.7976703",
        "citation_count": 22,
        "journal_if": 1.2,
        "summary_line": "Summary fetched for the related journal work.",
        "summary_source": "abstract",
        "summary_provider": "openalex",
    }
    monkeypatch.setattr(mq, "enrich_citation_detail_meta", lambda _seed: dict(detail))

    quality = mq.citation_metadata_quality(detail)
    acceptance = mq.citation_metadata_export_acceptance({**detail, "metadata_quality": quality})
    repaired = mq.repair_citation_metadata_item(detail)
    meta = repaired["meta"]

    issue_codes = {item["code"] for item in quality["issues"]}
    assert quality["status"] == "error"
    assert "raw_reference_title_conflict" in issue_codes
    assert "raw_reference_doi_identity_conflict" in issue_codes
    assert "raw_reference_venue_conflict" in issue_codes
    assert acceptance["export_ready"] is False
    assert meta["title"] == "The missing cone problem and low-pass distortion in optical serial sectioning microscopy"
    assert meta["authors"].startswith("Macias-Garza, F., Bovik, A. C.")
    assert meta["venue"] == "IEEE Trans. Acoust"
    assert meta["year"] == "1988"
    assert "doi" not in meta
    assert "doi_url" not in meta
    assert "citation_count" not in meta
    assert "journal_if" not in meta
    assert "summary_line" not in meta
    assert repaired["export_acceptance"]["export_ready"] is True
    assert repaired["export_acceptance"]["export_mode"] == "complete_bibliography"


def test_repair_rejects_inferred_doi_when_visible_title_was_already_recovered(monkeypatch) -> None:
    raw = (
        "Macias-Garza, F., Bovik, A. C., Diller, K. R., Aggarwal, S. J. & Aggarwal, J. K. "
        "The missing cone problem and low-pass distortion in optical serial sectioning microscopy. "
        "IEEE Trans. Acoust., Speech, Signal Process. 2, 890-893 (1988)."
    )
    detail = {
        "source_path": "citing-paper.en.md",
        "citation_route": "system_b",
        "is_inpaper": True,
        "raw": raw,
        "title": "The missing cone problem and low-pass distortion in optical serial sectioning microscopy",
        "authors": "Macias-Garza, F., Bovik, A. C., Diller, K. R., Aggarwal, S. J. & Aggarwal, J. K",
        "venue": "IEEE Trans. Acoust., Speech, Signal Process.",
        "year": "1988",
        "doi": "10.1117/12.7976703",
        "doiUrl": "https://doi.org/10.1117/12.7976703",
        "citationCount": 22,
        "journalIf": 1.2,
        "journalQuartile": "Q4",
    }
    monkeypatch.setattr(
        mq,
        "fetch_crossref_work_by_doi_status",
        lambda _doi: (
            {"title": ["Missing Cone Of Frequencies And Low-Pass Distortion In Three-Dimensional Microscopic Images"]},
            "ready",
        ),
    )
    monkeypatch.setattr(mq, "enrich_citation_detail_meta", lambda seed: dict(seed))

    result = mq.repair_citation_metadata_item(detail)
    meta = result["meta"]

    assert result["changed_fields"] == ["doi", "doi_url", "citation_count", "journal_if", "journal_quartile"]
    assert meta["title"] == detail["title"]
    assert meta["raw_reference_rejected_doi"] == "10.1117/12.7976703"
    assert "doi" not in meta
    assert "citation_count" not in meta
    assert "journal_if" not in meta
    assert "journal_quartile" not in meta
    assert result["export_acceptance"]["export_ready"] is True

    second_rejection = mq.verify_inferred_doi_against_raw_reference(
        {**meta, "doi": "10.1145/1073204.1073257"}
    )
    assert "doi" not in second_rejection
    assert set(second_rejection["raw_reference_rejected_dois"]) == {
        "10.1117/12.7976703",
        "10.1145/1073204.1073257",
    }


def test_repair_rechecks_replacement_doi_after_transient_registry_failure(monkeypatch) -> None:
    raw = (
        "Macias-Garza, F., Bovik, A. C., Diller, K. R., Aggarwal, S. J. & Aggarwal, J. K. "
        "The missing cone problem and low-pass distortion in optical serial sectioning microscopy. "
        "IEEE Trans. Acoust., Speech, Signal Process. 2, 890-893 (1988)."
    )
    detail = {
        "source_path": "citing-paper.en.md",
        "citation_route": "system_b",
        "is_inpaper": True,
        "raw": raw,
        "title": "The missing cone problem and low-pass distortion in optical serial sectioning microscopy",
        "authors": "Macias-Garza, F., Bovik, A. C., Diller, K. R., Aggarwal, S. J. & Aggarwal, J. K",
        "venue": "IEEE Trans. Acoust., Speech, Signal Process.",
        "year": "1988",
        "doi": "10.1117/12.7976703",
    }
    alternate_doi = "10.1145/1073204.1073257"
    registry_calls = iter(
        [
            (
                {"title": ["Missing Cone Of Frequencies And Low-Pass Distortion In Three-Dimensional Microscopic Images"]},
                "ready",
            ),
            (None, "error"),
            ({"title": ["Dual photography"]}, "ready"),
        ]
    )
    monkeypatch.setattr(mq, "fetch_crossref_work_by_doi_status", lambda _doi: next(registry_calls))
    monkeypatch.setattr(
        mq,
        "hydrate_repaired_citation_metadata",
        lambda seed, **_kwargs: {**seed, "doi": alternate_doi, "doi_url": f"https://doi.org/{alternate_doi}"},
    )

    result = mq.repair_citation_metadata_item(detail)
    meta = result["meta"]

    assert "doi" not in meta
    assert "doi_url" not in meta
    assert set(meta["raw_reference_rejected_dois"]) == {
        "10.1117/12.7976703",
        alternate_doi,
    }
    assert result["export_acceptance"]["export_ready"] is True


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


def test_system_a_doi_identity_is_exportable_without_inventing_bibliographic_fields() -> None:
    detail = {
        "citation_route": "system_a",
        "is_inpaper": False,
        "source_path": "db/NatPhoton-2025-Structured detection/Structured detection.en.md",
        "source_name": "NatPhoton-2025-Structured detection.pdf",
        "bibliographic_title": "Structured detection",
        "title": "Structured detection",
        "heading_path": "Structured detection / Abstract",
        "authors": "",
        "venue": "",
        "year": "",
        "doi": "10.1038/example.structured",
    }

    acceptance = mq.citation_metadata_export_acceptance(detail)

    assert acceptance["export_ready"] is True
    assert acceptance["export_mode"] == "system_a_doi"
    assert acceptance["field_ready"]["authors"] is False
    assert acceptance["field_ready"]["venue"] is False
    assert acceptance["field_ready"]["year"] is False


def test_system_a_local_source_export_refuses_section_heading_as_article_title() -> None:
    detail = {
        "citation_route": "system_a",
        "is_inpaper": False,
        "source_path": "db/paper/paper.en.md",
        "source_name": "paper.pdf",
        "title": "3. Results",
        "heading_path": "3. Results",
    }

    acceptance = mq.citation_metadata_export_acceptance(detail)

    assert acceptance["export_ready"] is False
    assert acceptance["export_mode"] == ""


def test_complete_bibliography_is_exportable_without_a_doi() -> None:
    detail = {
        "source_path": "refs/classic-paper.en.md",
        "title": "A classic paper without a DOI",
        "authors": "Ada Researcher",
        "venue": "Journal of Archival Results",
        "year": "1988",
    }

    acceptance = mq.citation_metadata_export_acceptance(detail)

    assert acceptance["export_ready"] is True
    assert acceptance["export_mode"] == "complete_bibliography"
    assert acceptance["field_ready"]["doi"] is False


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
