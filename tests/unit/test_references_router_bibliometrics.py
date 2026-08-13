from __future__ import annotations

from pathlib import Path

from api.routers import references
from kb.citation_card import compose_citation_card


def test_source_bound_system_a_bibliometrics_uses_source_metadata_not_ref_number(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source_path = str(tmp_path / "source-paper.en.md")
    calls: list[str] = []

    def source_meta(**kwargs):
        calls.append(str(kwargs.get("source_path") or ""))
        return {
            "title": "The Actual Source Paper",
            "authors": ["Source Author"],
            "venue": "Source Journal",
            "year": "2025",
            "doi": "10.1000/source-paper",
            "citation_count": 17,
            "citation_source": "openalex",
            "journal_if": 4.2,
            "journal_if_source": "jcr",
            "summary_line": "This is the abstract of the actual source paper.",
            "summary_source": "abstract",
            "summary_provider": "local_markdown",
            "summary_generation": "extractive_local_markdown",
            "summary_locale": "en",
            "summary_quality": {
                "ok": True,
                "status": "grounded",
                "locale": "en",
            },
        }

    def forbidden(*_args, **_kwargs):
        raise AssertionError("source-bound System A must bypass generic bibliography enrichment")

    monkeypatch.setattr(references, "ensure_source_citation_meta", source_meta)
    monkeypatch.setattr(references, "_source_reference_index_identity_meta", lambda _value: {})
    monkeypatch.setattr(references, "_resolve_public_reference_source_input", lambda value: value)
    monkeypatch.setattr(references, "_pdf_dir", lambda: tmp_path)
    monkeypatch.setattr(references, "_md_dir", lambda: tmp_path)
    monkeypatch.setattr(references, "_lib_store", lambda: object())
    monkeypatch.setattr(references, "_prepare_bibliometrics_identity", forbidden)
    monkeypatch.setattr(references, "hydrate_repaired_citation_metadata", forbidden)
    monkeypatch.setattr(references, "enrich_citation_detail_meta", forbidden)
    monkeypatch.setattr(references, "persist_repaired_citation_metadata", forbidden)

    result = references.get_bibliometrics(
        references.BibliometricsBody(
            target_locale="en",
            meta={
                "citation_route": "system_a",
                "is_inpaper": False,
                "source_path": source_path,
                "source_name": "source-paper.pdf",
                "num": 1,
                "linked_nums": [1],
                "raw": "The local evidence cites an unrelated bibliography entry [1].",
                "title": "Wrong Reference One",
                "authors": ["Wrong Author"],
                "venue": "Wrong Venue",
                "year": "1999",
                "doi": "10.1000/wrong-reference-one",
                "citation_count": 999,
                "journal_if": 99.9,
            },
        )
    )

    assert calls == [source_path]
    assert result["title"] == "The Actual Source Paper"
    assert result["authors"] == ["Source Author"]
    assert result["venue"] == "Source Journal"
    assert result["year"] == "2025"
    assert result["doi"] == "10.1000/source-paper"
    assert result["citation_count"] == 17
    assert result["journal_if"] == 4.2
    assert result["summary_line"] == "This is the abstract of the actual source paper."
    assert result["bibliometrics_identity_source"] == "source_path"
    assert result["source_metadata_status"] == "ready"
    assert result["citation_route"] == "system_a"
    assert result["is_inpaper"] is False
    assert result["source_path"] == source_path
    assert "num" not in result
    assert "linked_nums" not in result
    assert "raw" not in result


def test_source_bound_system_a_bibliometrics_fails_without_bibliography_fallback(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source_path = str(tmp_path / "missing-source.en.md")

    def source_meta(**_kwargs):
        raise RuntimeError("source metadata unavailable")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("unsafe generic bibliography fallback was called")

    monkeypatch.setattr(references, "ensure_source_citation_meta", source_meta)
    monkeypatch.setattr(references, "_source_reference_index_identity_meta", lambda _value: {})
    monkeypatch.setattr(references, "_resolve_public_reference_source_input", lambda value: value)
    monkeypatch.setattr(references, "_pdf_dir", lambda: tmp_path)
    monkeypatch.setattr(references, "_md_dir", lambda: tmp_path)
    monkeypatch.setattr(references, "_lib_store", lambda: object())
    monkeypatch.setattr(references, "_local_source_summary_meta", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(references, "_prepare_bibliometrics_identity", forbidden)
    monkeypatch.setattr(references, "hydrate_repaired_citation_metadata", forbidden)
    monkeypatch.setattr(references, "enrich_citation_detail_meta", forbidden)
    monkeypatch.setattr(references, "persist_repaired_citation_metadata", forbidden)

    result = references.get_bibliometrics(
        references.BibliometricsBody(
            target_locale="zh",
            meta={
                "citation_route": "system_a",
                "is_inpaper": False,
                "source_path": source_path,
                "source_name": "missing-source.pdf",
                "num": 1,
                "linked_nums": [1],
                "raw": "Evidence text ending in citation [1].",
                "title": "Wrong Upstream Reference",
                "authors": ["Wrong Author"],
                "venue": "Wrong Journal",
                "year": "2020",
                "doi": "10.1000/wrong-upstream",
                "citation_count": 500,
                "journal_if": 50.0,
            },
        )
    )

    assert result["source_metadata_status"] == "unavailable"
    assert result["bibliometrics_identity_source"] == "source_path"
    assert result["bibliometrics_checked"] is True
    assert result["citation_route"] == "system_a"
    assert result["is_inpaper"] is False
    assert result["source_path"] == source_path
    for key in (
        "title",
        "authors",
        "venue",
        "year",
        "doi",
        "doi_url",
        "citation_count",
        "journal_if",
        "num",
        "linked_nums",
        "raw",
    ):
        assert key not in result


def test_source_bound_system_a_bibliometrics_prefers_current_indexed_source_doi(
    monkeypatch,
    tmp_path: Path,
) -> None:
    source_path = str(tmp_path / "informer.en.md")

    monkeypatch.setattr(
        references,
        "ensure_source_citation_meta",
        lambda **_kwargs: {
            "title": "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting",
            "summary_line": "Informer introduces ProbSparse attention for long-sequence forecasting.",
            "summary_source": "abstract",
            "summary_provider": "local_markdown",
            "summary_locale": "en",
            "summary_quality": {"ok": True, "status": "grounded", "locale": "en"},
        },
    )
    monkeypatch.setattr(
        references,
        "_source_reference_index_identity_meta",
        lambda _value: {
            "doi": "10.1609/aaai.v35i12.17325",
            "doi_url": "https://doi.org/10.1609/aaai.v35i12.17325",
            "doi_identity_source": "source_reference_index",
            "source_reference_lookup_version": references.REFERENCE_LOOKUP_VERSION,
            "metadata_repair_status": "repaired",
            "metadata_changed_fields": ["doi"],
        },
    )
    monkeypatch.setattr(references, "_resolve_public_reference_source_input", lambda value: value)
    monkeypatch.setattr(references, "_pdf_dir", lambda: tmp_path)
    monkeypatch.setattr(references, "_md_dir", lambda: tmp_path)
    monkeypatch.setattr(references, "_lib_store", lambda: object())

    result = references.get_bibliometrics(
        references.BibliometricsBody(
            target_locale="en",
            meta={
                "citation_route": "system_a",
                "is_inpaper": False,
                "source_path": source_path,
                "title": "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting",
                "doi": "10.18653/v1/d15-1166",
            },
        )
    )

    assert result["doi"] == "10.1609/aaai.v35i12.17325"
    assert result["doi_url"] == "https://doi.org/10.1609/aaai.v35i12.17325"
    assert result["doi_identity_source"] == "source_reference_index"
    assert result["source_reference_lookup_version"] == references.REFERENCE_LOOKUP_VERSION
    assert result["metadata_repair_status"] == "repaired"
    assert result["metadata_changed_fields"] == ["doi"]


def test_bibliometrics_rejects_local_source_summary_for_reader_reference() -> None:
    meta = {
        "is_inpaper": True,
        "citation_context_source": "reader_references",
        "binding_status": "reader_reference",
        "source_path": "opened-paper.en.md",
        "title": "On the experimental verification of quantum complexity in linear optics",
        "doi": "10.1038/nphoton.2014.152",
        "summary_line": "High-resolution single-photon imaging remains a big challenge.",
        "summary_source": "abstract",
        "summary_provider": "local_markdown",
        "summary_generation": "extractive_local_markdown",
        "summary_quality": {
            "ok": True,
            "status": "grounded",
            "provider": "local_markdown",
            "generation": "extractive_local_markdown",
        },
    }

    stripped = references._strip_misbound_local_source_summary(meta)

    assert "summary_line" not in stripped
    assert "summary_source" not in stripped
    assert stripped["title"] == meta["title"]
    assert stripped["doi"] == meta["doi"]
    assert references._bibliometrics_accept_local_source_summary(meta) is False


def test_bibliometrics_allows_local_summary_for_direct_source_item() -> None:
    meta = {
        "is_inpaper": False,
        "source_path": "opened-paper.en.md",
        "source_name": "Opened Paper.pdf",
    }

    assert references._bibliometrics_accept_local_source_summary(meta) is True


def test_bibliometrics_strips_reader_context_summary_and_stale_acceptance() -> None:
    meta = {
        "is_inpaper": True,
        "citation_context_source": "reader_occurrence",
        "binding_status": "reader_reference",
        "title": "Single-photon avalanche diode imagers in biophotonics: review and outlook",
        "doi": "10.1038/s41377-019-0191-5",
        "summary_line": "The opened paper cites this upstream work as reference [1].",
        "summary_source": "reader_reference_link",
        "metadata_export_acceptance": {
            "summary_export_ready": True,
            "summary": {
                "export_ready": True,
                "source": "reader_reference_link",
            },
        },
    }

    stripped = references._strip_misbound_local_source_summary(meta)

    assert "summary_line" not in stripped
    assert "summary_source" not in stripped
    assert "metadata_export_acceptance" not in stripped
    assert stripped["doi"] == meta["doi"]


def test_system_b_card_preserves_reader_reference_summary_source() -> None:
    detail = compose_citation_card(
        {
            "is_inpaper": True,
            "source_name": "Opened Paper.pdf",
            "title": "Silicon single-photon avalanche diodes with nano-structured light trapping",
            "raw": "[14] Zang, K. et al. Silicon single-photon avalanche diodes with nanostructured light trapping.",
            "citation_context": "This bibliography entry is linked from the opened Reader document.",
            "citation_context_source": "reader_reference_link",
            "summary_line": "This bibliography entry is linked from the opened Reader document.",
            "summary_source": "reader_reference_link",
            "summary_provider": "reader",
            "summary_quality": {"ok": False, "status": "context_only"},
        },
        locale="en",
    )

    assert detail["summary_line"] == "This bibliography entry is linked from the opened Reader document."
    assert detail["summary_source"] == "reader_reference_link"
    assert detail["summary_provider"] == "reader"
    assert detail["summary_quality"]["status"] == "context_only"


def test_reader_reference_cite_details_links_cited_missing_reference_entry(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(references, "_reader_reference_index_data", lambda: {})
    md_text = (
        "# Demo\n\n"
        "Author A [1,2]\n\n"
        "## Abstract\n\n"
        "The introduction cites a continuous reference range [1-3].\n\n"
        "## References\n\n"
        "[1] Alpha, A. First reference. Journal 1, 1-2 (2020).\n"
        "[3] Gamma, G. Third reference. Journal 3, 3-4 (2022).\n"
    )

    cards = references._reader_reference_cite_details(
        md_text,
        source_path=str(tmp_path / "demo.en.md"),
        source_name="demo.en.md",
        md_path=tmp_path / "demo.en.md",
        doc_hash="abc123",
    )

    by_num = {int(card.get("num") or 0): card for card in cards}
    assert sorted(by_num) == [1, 2, 3]
    missing = by_num[2]
    assert missing["anchor"].endswith("-2")
    assert missing["linked_nums"] == [2]
    assert missing["binding_status"] == "missing_reference_entry"
    assert missing["bibliometrics_checked"] is True
    assert "missing_reference_entry" in missing["card_quality_flags"]
    assert "does not contain a matching bibliography entry" in missing["card_reference_entry"]
