from __future__ import annotations

from api import reference_external_abstracts as abstracts


def test_summary_from_crossref_abstract_uses_injected_fetcher() -> None:
    out = abstracts._summary_from_crossref_abstract(
        {"doi": "https://doi.org/10.1000/demo"},
        fetch_crossref_work_by_doi=lambda doi: {
            "abstract": (
                "<jats:p>We propose an adaptive imaging method for low-light reconstruction. "
                "Experiments show improved fidelity under limited measurements.</jats:p>"
            )
        },
    )

    assert "adaptive imaging method" in out
    assert "<jats" not in out


def test_summary_from_openalex_abstract_uses_injected_fetcher() -> None:
    out = abstracts._summary_from_openalex_abstract(
        {"doi": "10.1000/openalex"},
        openalex_work_by_doi=lambda doi: {
            "abstract_inverted_index": {
                "We": [0],
                "improve": [1],
                "single-pixel": [2],
                "imaging": [3],
                "reconstruction": [4],
                "quality.": [5],
            }
        },
    )

    assert out == "We improve single-pixel imaging reconstruction quality."


def test_summary_from_semantic_scholar_rejects_doi_mismatch() -> None:
    out = abstracts._summary_from_semantic_scholar_abstract(
        {"doi": "10.1000/expected", "title": "Adaptive sampling for imaging"},
        semantic_scholar_paper_by_doi=lambda doi: {
            "title": "Adaptive sampling for imaging",
            "externalIds": {"DOI": "10.1000/other"},
            "abstract": "This abstract should not be accepted.",
        },
        title_similarity=lambda left, right: 1.0,
    )

    assert out == ""


def test_summary_from_semantic_scholar_accepts_valid_abstract() -> None:
    abstract = (
        "We develop an adaptive sampling method for single-pixel imaging that selects informative "
        "illumination patterns during acquisition. Experiments show that the strategy improves "
        "reconstruction quality under limited measurements and low-light conditions."
    )
    out = abstracts._summary_from_semantic_scholar_abstract(
        {"doi": "10.1000/semantic", "title": "Adaptive sampling for single-pixel imaging"},
        semantic_scholar_paper_by_doi=lambda doi: {
            "title": "Adaptive sampling for single-pixel imaging",
            "externalIds": {"DOI": "10.1000/semantic"},
            "abstract": abstract,
        },
        title_similarity=lambda left, right: 1.0,
    )

    assert "adaptive sampling method" in out
    assert "limited measurements" in out


def test_valid_external_abstract_candidate_rejects_landing_page_boilerplate() -> None:
    out = abstracts._valid_external_abstract_candidate(
        "Access through your institution. Sign in to access this article navigation page.",
        title="A real paper",
    )

    assert out == ""


def test_doi_landing_page_abstract_reads_html_meta(monkeypatch) -> None:
    abstracts._doi_landing_page_abstract.cache_clear()
    calls: list[str] = []

    class Response:
        status_code = 200
        headers = {"content-type": "text/html; charset=utf-8"}
        text = '<meta name="citation_abstract" content="A concise abstract from a DOI landing page.">'

    def fake_get(url: str, **kwargs):
        calls.append(url)
        assert kwargs["allow_redirects"] is True
        return Response()

    monkeypatch.setattr(abstracts.requests, "get", fake_get)

    assert abstracts._doi_landing_page_abstract("10.1000/demo") == "A concise abstract from a DOI landing page."
    assert calls == ["https://doi.org/10.1000/demo"]


def test_summary_from_doi_landing_page_uses_injected_fetcher() -> None:
    out = abstracts._summary_from_doi_landing_page(
        {"doi": "10.1000/landing", "title": "Adaptive sampling for single-pixel imaging"},
        doi_landing_page_abstract=lambda doi: (
            "We analyze adaptive sampling for single-pixel imaging and compare reconstruction "
            "quality under limited measurements. Results show improved image fidelity in "
            "low-light acquisition settings."
        ),
    )

    assert "adaptive sampling" in out
