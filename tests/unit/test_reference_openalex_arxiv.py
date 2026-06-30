from __future__ import annotations

from api import reference_openalex_arxiv as openalex_arxiv


def test_normalize_title_for_openalex_search_collapses_whitespace_and_caps_length() -> None:
    raw = "  Neural   reflectance\nfields for appearance acquisition  " + ("x" * 300)

    out = openalex_arxiv._normalize_title_for_openalex_search(raw)

    assert "\n" not in out
    assert "  " not in out
    assert len(out) == 240


def test_title_similarity_for_openalex_combines_sequence_and_token_overlap() -> None:
    close = openalex_arxiv._title_similarity_for_openalex(
        "Neural reflectance fields for appearance acquisition",
        "Neural Reflectance Fields for Appearance Acquisition",
    )
    far = openalex_arxiv._title_similarity_for_openalex(
        "Neural reflectance fields for appearance acquisition",
        "Single-shot compressive spectral imaging",
    )

    assert close > 0.95
    assert far < close


def test_should_try_openalex_arxiv_title_uses_raw_or_venue_signal() -> None:
    assert openalex_arxiv._should_try_openalex_arxiv_title(
        {"title": "Neural reflectance fields for appearance acquisition", "venue": "Conference"},
        raw="arXiv:2008.03824",
    )
    assert openalex_arxiv._should_try_openalex_arxiv_title(
        {"title": "Neural reflectance fields for appearance acquisition", "venue": "arXiv preprint"},
        raw="",
    )
    assert not openalex_arxiv._should_try_openalex_arxiv_title({"title": "Short"}, raw="arXiv")


def test_openalex_arxiv_meta_by_title_picks_matching_arxiv_result(monkeypatch) -> None:
    calls: list[dict] = []

    class Response:
        status_code = 200

        def json(self):
            return {
                "results": [
                    {
                        "title": "Unrelated conference paper",
                        "doi": "https://doi.org/10.5555/not-arxiv",
                    },
                    {
                        "title": "Neural reflectance fields for appearance acquisition",
                        "doi": "https://doi.org/10.48550/arXiv.2008.03824",
                        "publication_year": 2020,
                        "primary_location": {"source": {"display_name": "arXiv"}},
                    },
                ]
            }

    def fake_get(url: str, **kwargs):
        calls.append({"url": url, **kwargs})
        return Response()

    monkeypatch.setattr(openalex_arxiv.requests, "get", fake_get)

    out = openalex_arxiv._openalex_arxiv_meta_by_title("Neural reflectance fields for appearance acquisition")

    assert out["doi"] == "10.48550/arxiv.2008.03824"
    assert out["doi_url"] == "https://doi.org/10.48550/arxiv.2008.03824"
    assert out["year"] == "2020"
    assert out["venue"] == "arXiv"
    assert out["match_method"] == "openalex_title_arxiv"
    assert calls[0]["url"] == "https://api.openalex.org/works"
    assert calls[0]["params"]["per-page"] == 8
