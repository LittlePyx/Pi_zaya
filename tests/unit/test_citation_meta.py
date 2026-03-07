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
