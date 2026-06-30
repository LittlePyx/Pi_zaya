from __future__ import annotations

from api import reference_detail_seed as seed


def test_detail_raw_seed_prefers_raw_then_card_entry() -> None:
    assert seed.detail_raw_seed({"raw": "raw text", "card_reference_entry": "card text"}) == "raw text"
    assert seed.detail_raw_seed({"card_reference_entry": "card text"}) == "card text"


def test_seed_detail_raw_fields_sets_raw_cite_fmt_and_doi() -> None:
    out = seed.seed_detail_raw_fields(
        {},
        raw="Gehm M. Demo paper. doi:10.1000/demo",
        normalize_doi_like=lambda value: "",
        extract_first_doi=lambda raw: "10.1000/demo",
        build_doi_url=lambda doi: f"https://doi.org/{doi}",
    )

    assert out["raw"].startswith("Gehm M")
    assert out["cite_fmt"].startswith("Gehm M")
    assert out["doi"] == "10.1000/demo"
    assert out["doi_url"] == "https://doi.org/10.1000/demo"


def test_seed_detail_raw_fields_does_not_override_existing_doi() -> None:
    out = seed.seed_detail_raw_fields(
        {"doi": "10.1000/existing", "raw": "existing raw"},
        raw="new raw doi:10.1000/new",
        normalize_doi_like=lambda value: "10.1000/existing",
        extract_first_doi=lambda raw: "10.1000/new",
        build_doi_url=lambda doi: f"https://doi.org/{doi}",
    )

    assert out["raw"] == "existing raw"
    assert out["doi"] == "10.1000/existing"
    assert "doi_url" not in out


def test_fallback_parse_raw_reference_uses_shared_parser() -> None:
    out = seed.fallback_parse_raw_reference(
        "[24] Gehm M, Brady D. Single-shot compressive spectral imaging. Optics Express, 2007.",
        meta={},
        arxiv_backfill_meta_from_texts=lambda *texts: {},
        fallback_fill_reference_meta_from_raw=lambda meta: {
            "authors": "Gehm M, Brady D",
            "title": "Single-shot compressive spectral imaging",
            "venue": "Optics Express",
            "year": "2007",
        },
    )

    assert out["authors"] == "Gehm M, Brady D"
    assert out["title"] == "Single-shot compressive spectral imaging"
    assert out["venue"] == "Optics Express"
    assert out["year"] == "2007"


def test_fallback_parse_raw_reference_handles_et_al_pattern() -> None:
    out = seed.fallback_parse_raw_reference(
        "Gehm et al. Single-shot compressive spectral imaging. Optics Express, 2007.",
        meta={},
        arxiv_backfill_meta_from_texts=lambda *texts: {},
        fallback_fill_reference_meta_from_raw=lambda meta: {},
    )

    assert out["authors"] == "Gehm et al"
    assert out["title"] == "Single-shot compressive spectral imaging"
    assert out["venue"] == "Optics Express, 2007"


def test_apply_raw_reference_fallback_preserves_existing_fields() -> None:
    out = seed.apply_raw_reference_fallback(
        {"title": "Existing title"},
        raw="Gehm M. Parsed title. Journal, 2024.",
        arxiv_backfill_meta_from_texts=lambda *texts: {},
        fallback_fill_reference_meta_from_raw=lambda meta: {
            "authors": "Gehm M",
            "title": "Parsed title",
            "venue": "Journal",
            "year": "2024",
        },
    )

    assert out["title"] == "Existing title"
    assert out["authors"] == "Gehm M"
    assert out["venue"] == "Journal"
    assert out["year"] == "2024"
