from __future__ import annotations

from api import reference_detail_crossref as crossref


def _merge(base: dict, incoming: dict) -> dict:
    return {**dict(base or {}), **dict(incoming or {})}


def test_merge_canonical_for_existing_doi_adds_doi_url_and_metadata() -> None:
    calls: list[dict] = []

    def fetch(**kwargs):
        calls.append(kwargs)
        return {
            "title": "Canonical title",
            "authors": "Gehm M, Brady D",
            "doi": "10.1000/demo",
        }

    out = crossref.merge_canonical_for_existing_doi(
        {"doi": "10.1000/demo", "title": "Local title"},
        title="Local title",
        venue="Optics Express",
        year="2007",
        doi="10.1000/demo",
        fetch_best_crossref_meta=fetch,
        is_weak_meta_value=lambda key, value: False,
        normalize_doi_like=lambda value: str(value or "").replace("https://doi.org/", "").lower(),
        merge_meta_prefer_richer=_merge,
        build_doi_url=lambda doi: f"https://doi.org/{doi}",
    )

    assert out["title"] == "Canonical title"
    assert out["authors"] == "Gehm M, Brady D"
    assert out["doi_url"] == "https://doi.org/10.1000/demo"
    assert calls[0]["doi_hint"] == "10.1000/demo"
    assert calls[0]["allow_title_only"] is False


def test_merge_reference_text_crossref_fetches_reference_then_canonical() -> None:
    best_calls: list[dict] = []

    def fetch_ref(**kwargs):
        return {"doi": "10.1000/ref", "title": "Reference title"}

    def fetch_best(**kwargs):
        best_calls.append(kwargs)
        return {"doi": "10.1000/ref", "venue": "Optics Express"}

    out = crossref.merge_reference_text_crossref(
        {"title": "Seed title"},
        raw="[1] Demo reference. doi:10.1000/ref",
        title="Seed title",
        venue="",
        year="",
        fetch_best_crossref_for_reference=fetch_ref,
        fetch_best_crossref_meta=fetch_best,
        is_weak_meta_value=lambda key, value: False,
        merge_meta_prefer_richer=_merge,
        build_doi_url=lambda doi: f"https://doi.org/{doi}",
    )

    assert out["doi"] == "10.1000/ref"
    assert out["doi_url"] == "https://doi.org/10.1000/ref"
    assert out["venue"] == "Optics Express"
    assert best_calls[0]["doi_hint"] == "10.1000/ref"


def test_merge_reference_text_crossref_noops_without_raw_match() -> None:
    out = crossref.merge_reference_text_crossref(
        {"title": "Seed title"},
        raw="",
        title="Seed title",
        venue="",
        year="",
        fetch_best_crossref_for_reference=lambda **kwargs: None,
        fetch_best_crossref_meta=lambda **kwargs: None,
        is_weak_meta_value=lambda key, value: False,
        merge_meta_prefer_richer=_merge,
        build_doi_url=lambda doi: f"https://doi.org/{doi}",
    )

    assert out == {"title": "Seed title"}


def test_merge_title_crossref_uses_title_only_fallback_when_primary_empty() -> None:
    calls: list[dict] = []

    def fetch_best(**kwargs):
        calls.append(kwargs)
        return {"title": "Recovered title", "doi": "10.1000/title"}

    out = crossref.merge_title_crossref(
        {"title": "Recovered title"},
        title="Recovered title",
        raw="",
        venue="",
        year="",
        fetch_crossref_meta=lambda *args, **kwargs: None,
        fetch_best_crossref_meta=fetch_best,
        is_weak_meta_value=lambda key, value: False,
        merge_meta_prefer_richer=_merge,
        build_doi_url=lambda doi: f"https://doi.org/{doi}",
    )

    assert out["doi"] == "10.1000/title"
    assert out["doi_url"] == "https://doi.org/10.1000/title"
    assert calls[0]["allow_title_only"] is True


def test_merge_title_crossref_uses_raw_as_search_title_when_title_missing() -> None:
    seen: list[str] = []

    def fetch(search_title: str, **kwargs):
        seen.append(search_title)
        return {"doi": "10.1000/raw"}

    out = crossref.merge_title_crossref(
        {},
        title="",
        raw="[24] Raw title from reference. Journal, 2024.",
        venue="",
        year="",
        fetch_crossref_meta=fetch,
        fetch_best_crossref_meta=lambda **kwargs: None,
        is_weak_meta_value=lambda key, value: False,
        merge_meta_prefer_richer=_merge,
        build_doi_url=lambda doi: f"https://doi.org/{doi}",
    )

    assert seen == ["Raw title from reference. Journal, 2024."]
    assert out["doi"] == "10.1000/raw"
    assert out["doi_url"] == "https://doi.org/10.1000/raw"
