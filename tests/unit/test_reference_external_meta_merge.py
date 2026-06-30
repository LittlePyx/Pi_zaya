from __future__ import annotations

from api import reference_external_meta_merge as merge


def test_external_meta_merge_mode_detects_doi_conflict() -> None:
    mode, reason, similarity = merge._external_meta_merge_mode(
        {"doi": "10.1000/correct"},
        {"doi": "10.2000/wrong", "title": "Wrong external paper"},
    )

    assert mode == "conflict"
    assert "DOI" in reason
    assert similarity == 0.0


def test_merge_meta_prefer_richer_stores_low_similarity_candidate_without_rewriting_identity() -> None:
    out = merge._merge_meta_prefer_richer(
        {
            "title": "Single-shot compressive spectral imaging",
            "authors": "Gehm M, Brady D",
            "venue": "Optics Express",
            "year": "2007",
        },
        {
            "title": "Completely unrelated external paper",
            "authors": "Wrong Author",
            "venue": "Wrong Journal",
            "year": "2024",
            "doi": "10.1000/external",
            "doi_url": "https://doi.org/10.1000/external",
            "citation_count": 50,
            "match_method": "bibliographic",
            "title_similarity": 0.25,
        },
    )

    assert out["title"] == "Single-shot compressive spectral imaging"
    assert out["authors"] == "Gehm M, Brady D"
    assert out["venue"] == "Optics Express"
    assert out["external_metadata_status"] == "candidate"
    assert out["external_title"] == "Completely unrelated external paper"
    assert out["doi"] == "10.1000/external"
    assert out["citation_count"] == 50


def test_merge_meta_prefer_richer_trusted_same_doi_updates_identity() -> None:
    out = merge._merge_meta_prefer_richer(
        {
            "title": "Reference 24",
            "authors": "",
            "doi": "10.1364/OE.15.014013",
        },
        {
            "title": "Single-shot compressive spectral imaging with a dual-disperser architecture",
            "authors": "Gehm M, Brady D",
            "venue": "Optics Express",
            "year": "2007",
            "doi": "10.1364/OE.15.014013",
            "title_similarity": 0.96,
        },
    )

    assert out["title"] == "Single-shot compressive spectral imaging with a dual-disperser architecture"
    assert out["authors"] == "Gehm M, Brady D"
    assert out["venue"] == "Optics Express"
    assert out["year"] == "2007"
    assert out["external_metadata_status"] == "trusted"
