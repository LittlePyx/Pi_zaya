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


def test_crossref_summary_bypasses_stale_none_with_status_fetcher() -> None:
    meta = {"doi": "10.1000/retry"}
    out = abstracts._summary_from_crossref_abstract(
        meta,
        fetch_crossref_work_by_doi=lambda doi: None,
        fetch_crossref_work_by_doi_status=lambda doi: (
            {"abstract": "We recover an abstract after a transient Crossref timeout."},
            "ready",
        ),
    )

    assert "recover an abstract" in out
    assert meta["summary_fetch_providers"]["crossref"] == "ready"


def test_crossref_summary_records_not_provided_separately_from_failure() -> None:
    meta = {"doi": "10.1000/no-abstract"}
    out = abstracts._summary_from_crossref_abstract(
        meta,
        fetch_crossref_work_by_doi=lambda doi: {"title": ["No abstract work"]},
        fetch_crossref_work_by_doi_status=lambda doi: (_ for _ in ()).throw(
            AssertionError("ready work should not be refetched")
        ),
    )

    assert out == ""
    assert meta["summary_fetch_providers"]["crossref"] == "not_provided"


def test_europe_pmc_summary_requires_exact_doi_and_matching_title() -> None:
    meta = {
        "doi": "10.1364/ao.56.004085",
        "title": "Single-pixel compressive diffractive imaging with structured illumination",
    }
    out = abstracts._summary_from_europe_pmc_abstract(
        meta,
        europe_pmc_work_by_doi=lambda doi: {
            "_kb_fetch_status": "ready",
            "doi": "10.1364/ao.56.004085",
            "title": "Single-pixel compressive diffractive imaging with structured illumination",
            "abstractText": (
                "We present a compressive diffractive imaging method using structured illumination. "
                "Experiments show accurate reconstruction from limited single-pixel measurements."
            ),
        },
        title_similarity=lambda left, right: 1.0,
    )

    assert "compressive diffractive imaging" in out
    assert meta["summary_fetch_providers"]["europe_pmc"] == "ready"


def test_europe_pmc_summary_rejects_doi_mismatch() -> None:
    meta = {"doi": "10.1000/expected", "title": "Expected work"}
    out = abstracts._summary_from_europe_pmc_abstract(
        meta,
        europe_pmc_work_by_doi=lambda doi: {
            "_kb_fetch_status": "ready",
            "doi": "10.1000/other",
            "title": "Expected work",
            "abstractText": "This abstract must not be attached to the expected work.",
        },
        title_similarity=lambda left, right: 1.0,
    )

    assert out == ""
    assert meta["summary_fetch_providers"]["europe_pmc"] == "identity_mismatch"


def test_europe_pmc_429_is_retryable_and_not_cached(monkeypatch) -> None:
    calls = 0

    class Response:
        def __init__(self, status_code: int):
            self.status_code = status_code

        def json(self):
            return {
                "resultList": {
                    "result": [
                        {
                            "doi": "10.1364/ao.56.004085",
                            "title": "Single-pixel compressive diffractive imaging",
                            "abstractText": "The abstract is returned after the rate limit clears.",
                        }
                    ]
                }
            }

    def fake_get(*args, **kwargs):
        nonlocal calls
        calls += 1
        assert kwargs["params"]["query"] == 'DOI:"10.1364/ao.56.004085"'
        return Response(429 if calls == 1 else 200)

    monkeypatch.setattr(abstracts.requests, "get", fake_get)

    first = abstracts._europe_pmc_work_by_doi("10.1364/ao.56.004085")
    second = abstracts._europe_pmc_work_by_doi("10.1364/ao.56.004085")

    assert first["_kb_fetch_status"] == "failed"
    assert first["_kb_http_status"] == 429
    assert second["_kb_fetch_status"] == "ready"
    assert second["abstractText"].startswith("The abstract is returned")
    assert calls == 2


def test_datacite_summary_uses_only_exact_doi_abstract_description() -> None:
    meta = {"doi": "10.5281/zenodo.11284050", "title": "Calibrated imaging dataset"}
    out = abstracts._summary_from_datacite_description(
        meta,
        datacite_doi_record=lambda doi: {
            "_kb_fetch_status": "ready",
            "doi": "10.5281/zenodo.11284050",
            "titles": [{"title": "Calibrated imaging dataset"}],
            "descriptions": [
                {"descriptionType": "TableOfContents", "description": "Files and folders"},
                {
                    "descriptionType": "Abstract",
                    "description": (
                        "We release a calibrated imaging dataset with acquisition metadata. "
                        "Validation experiments show consistent reconstruction quality."
                    ),
                },
            ],
        },
        title_similarity=lambda left, right: 1.0,
    )

    assert "calibrated imaging dataset" in out
    assert "Files and folders" not in out
    assert meta["summary_fetch_providers"]["datacite"] == "ready"


def test_datacite_is_skipped_for_crossref_registered_article() -> None:
    meta = {
        "doi": "10.1000/article",
        "summary_fetch_providers": {"crossref": "not_provided"},
    }
    out = abstracts._summary_from_datacite_description(
        meta,
        datacite_doi_record=lambda doi: (_ for _ in ()).throw(
            AssertionError("Crossref article should not query DataCite")
        ),
        title_similarity=lambda left, right: 1.0,
    )

    assert out == ""
    assert meta["summary_fetch_providers"]["datacite"] == "not_applicable"


def test_datacite_429_is_retryable_and_not_cached(monkeypatch) -> None:
    calls = 0

    class Response:
        def __init__(self, status_code: int):
            self.status_code = status_code

        def json(self):
            return {
                "data": {
                    "attributes": {
                        "doi": "10.5281/zenodo.11284050",
                        "descriptions": [
                            {
                                "descriptionType": "Abstract",
                                "description": "The dataset abstract is available after retry.",
                            }
                        ],
                    }
                }
            }

    def fake_get(url: str, **kwargs):
        nonlocal calls
        calls += 1
        assert url.endswith("10.5281%2Fzenodo.11284050")
        return Response(429 if calls == 1 else 200)

    monkeypatch.setattr(abstracts.requests, "get", fake_get)

    first = abstracts._datacite_doi_record("10.5281/zenodo.11284050")
    second = abstracts._datacite_doi_record("10.5281/zenodo.11284050")

    assert first["_kb_fetch_status"] == "failed"
    assert first["_kb_http_status"] == 429
    assert second["_kb_fetch_status"] == "ready"
    assert second["doi"] == "10.5281/zenodo.11284050"
    assert calls == 2


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


def test_openalex_legacy_none_does_not_claim_a_transient_failure() -> None:
    meta = {"doi": "10.1000/openalex-missing"}

    out = abstracts._summary_from_openalex_abstract(
        meta,
        openalex_work_by_doi=lambda doi: None,
    )

    assert out == ""
    assert meta["summary_fetch_providers"]["openalex"] == "not_provided"


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


def test_semantic_scholar_429_is_retryable_and_not_cached(monkeypatch) -> None:
    calls = 0

    class Response:
        def __init__(self, status_code: int):
            self.status_code = status_code

        def json(self):
            return {
                "title": "Adaptive sampling for single-pixel imaging",
                "abstract": "A valid abstract becomes available after the rate limit clears.",
                "externalIds": {"DOI": "10.1000/semantic-retry"},
            }

    def fake_get(*args, **kwargs):
        nonlocal calls
        calls += 1
        return Response(429 if calls == 1 else 200)

    monkeypatch.setattr(abstracts.requests, "get", fake_get)

    first = abstracts._semantic_scholar_paper_by_doi("10.1000/semantic-retry")
    second = abstracts._semantic_scholar_paper_by_doi("10.1000/semantic-retry")

    assert first["_kb_fetch_status"] == "failed"
    assert first["_kb_http_status"] == 429
    assert second["_kb_fetch_status"] == "ready"
    assert second["abstract"].startswith("A valid abstract")
    assert calls == 2


def test_valid_external_abstract_candidate_rejects_landing_page_boilerplate() -> None:
    out = abstracts._valid_external_abstract_candidate(
        "Access through your institution. Sign in to access this article navigation page.",
        title="A real paper",
    )

    assert out == ""


def test_doi_landing_page_abstract_reads_html_meta(monkeypatch) -> None:
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
