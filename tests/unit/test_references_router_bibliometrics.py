from __future__ import annotations

from pathlib import Path

from api.routers import references
from kb.citation_card import compose_citation_card


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
