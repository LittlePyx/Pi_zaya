from kb import citation_meta


def test_fetch_best_crossref_meta_prefers_candidate_with_matching_venue(monkeypatch):
    q = "Imaging biological tissue with high-throughput single-pixel compressive holography"

    monkeypatch.setattr(
        citation_meta,
        "_crossref_search_title_raw",
        lambda *_args, **_kwargs: [
            {
                "title": [q],
                "container-title": [],
                "issued": {"date-parts": [[2021]]},
                "author": [{"family": "Wu", "given": "Daixuan"}],
                "DOI": "10.21203/rs.3.rs-129598/v1",
            },
            {
                "title": [q],
                "container-title": ["Nature Communications"],
                "issued": {"date-parts": [[2021]]},
                "author": [{"family": "Wu", "given": "Daixuan"}],
                "DOI": "10.1038/s41467-021-24990-0",
            },
        ],
    )

    out = citation_meta.fetch_best_crossref_meta(
        query_title=q,
        expected_year="2021",
        expected_venue="NatCommun",
        allow_title_only=True,
        min_score=0.90,
    )

    assert isinstance(out, dict)
    assert str(out.get("doi") or "") == "10.1038/s41467-021-24990-0"


def test_fetch_best_openalex_meta_uses_title_year_author_gate(monkeypatch):
    q = "Optical imaging by means of two-photon quantum entanglement"

    monkeypatch.setattr(
        citation_meta,
        "_openalex_search_title_raw",
        lambda *_args, **_kwargs: [
            {
                "title": "Optical imaging by means of two-photon quantum entanglement",
                "publication_year": 1994,
                "doi": "https://doi.org/10.1000/wrong",
                "authorships": [{"author": {"display_name": "Alice Example"}}],
                "primary_location": {"source": {"display_name": "Physical Review A"}},
                "biblio": {},
            },
            {
                "title": "Optical imaging by means of two-photon quantum entanglement",
                "publication_year": 1995,
                "doi": "https://doi.org/10.1103/PhysRevA.52.R3429",
                "authorships": [{"author": {"display_name": "T. B. Pittman"}}],
                "primary_location": {"source": {"display_name": "Physical Review A"}},
                "biblio": {"volume": "52", "first_page": "R3429", "last_page": "R3432"},
            },
        ],
    )

    out = citation_meta.fetch_best_openalex_meta(
        query_title=q,
        reference_text=f"Pittman T B. {q}. Phys. Rev. A, 52:R3429-R3432, 1995.",
        min_score=0.90,
    )

    assert isinstance(out, dict)
    assert str(out.get("doi") or "") == "10.1103/PhysRevA.52.R3429"
    assert str(out.get("match_method") or "") == "openalex_title"


def test_clean_reference_query_expands_markdown_venue_aliases():
    raw = "[1] P. Kilcullen, T. Ozaki, J. Liang, *Nat. Commun.* **2022**, *13*, 7879."

    out = citation_meta._clean_reference_for_query(raw)

    assert "Nature Communications" in out
    assert "*" not in out
    assert "[1]" not in out
    assert citation_meta.extract_first_author_family_hint(citation_meta._reference_hint_text(raw)) == "kilcullen"


def test_fetch_best_crossref_for_reference_accepts_compact_bibliographic_match(monkeypatch):
    raw = "[1] P. Kilcullen, T. Ozaki, J. Liang, *Nat. Commun.* **2022**, *13*, 7879."

    monkeypatch.setattr(
        citation_meta,
        "_crossref_search_bibliographic_raw",
        lambda *_args, **_kwargs: [
            {
                "title": ["Unrelated single-pixel paper"],
                "container-title": ["Optics Express"],
                "issued": {"date-parts": [[2022]]},
                "author": [{"family": "Example", "given": "A."}],
                "volume": "30",
                "page": "1-8",
                "DOI": "10.1000/wrong",
            },
            {
                "title": ["Compressed ultrahigh-speed single-pixel imaging by swept aggregate patterns"],
                "container-title": ["Nature Communications"],
                "issued": {"date-parts": [[2022]]},
                "author": [
                    {"family": "Kilcullen", "given": "Peter"},
                    {"family": "Ozaki", "given": "T."},
                    {"family": "Liang", "given": "Jinyang"},
                ],
                "volume": "13",
                "page": "7879",
                "DOI": "10.1038/s41467-022-35585-8",
            },
        ],
    )

    out = citation_meta.fetch_best_crossref_for_reference(reference_text=raw)

    assert isinstance(out, dict)
    assert out["doi"] == "10.1038/s41467-022-35585-8"
    assert out["match_method"] == "bibliographic_compact"
    assert float(out["structured_match_score"]) >= 0.78
