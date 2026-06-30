from __future__ import annotations

from api import reference_detail_pipeline as pipeline


def _merge_meta(base: dict, incoming: dict) -> dict:
    out = dict(base or {})
    out.update({k: v for k, v in (incoming or {}).items() if v not in (None, "", [], {})})
    return out


def _norm_doi(value: str) -> str:
    return str(value or "").strip().lower().replace("https://doi.org/", "")


def test_enrich_citation_detail_meta_merges_existing_doi_canonical_then_summary() -> None:
    def fetch_best_crossref_meta(**kwargs):
        assert kwargs["doi_hint"] == "10.1000/demo"
        return {"doi": "10.1000/demo", "title": "Canonical title", "venue": "Optics Express"}

    def enrich_bibliometrics(meta: dict) -> dict:
        out = dict(meta)
        out["citation_count"] = "42"
        return out

    def ensure_summary_line(meta: dict, *, allow_crossref_abstract: bool) -> dict:
        assert allow_crossref_abstract is True
        out = dict(meta)
        out["summary_line"] = f"{out['title']} summary"
        return out

    out = pipeline.enrich_citation_detail_meta(
        {"doi": "10.1000/demo", "title": "Weak seed"},
        normalize_reference_for_popup=lambda detail: dict(detail),
        normalize_doi_like=_norm_doi,
        extract_first_doi=lambda raw: "",
        build_doi_url=lambda doi: f"https://doi.org/{doi}",
        arxiv_backfill_meta_from_texts=lambda *texts: {},
        fallback_fill_reference_meta_from_raw=lambda meta: {},
        merge_meta_prefer_richer=_merge_meta,
        fetch_best_crossref_meta=fetch_best_crossref_meta,
        fetch_best_crossref_for_reference=lambda **kwargs: None,
        fetch_crossref_meta=lambda *args, **kwargs: None,
        is_weak_meta_value=lambda key, value: False,
        should_try_openalex_arxiv_title=lambda meta, *, raw: False,
        openalex_arxiv_meta_by_title=lambda title: {},
        enrich_bibliometrics=enrich_bibliometrics,
        ensure_summary_line=ensure_summary_line,
    )

    assert out["title"] == "Canonical title"
    assert out["venue"] == "Optics Express"
    assert out["citation_count"] == "42"
    assert out["summary_line"] == "Canonical title summary"


def test_enrich_citation_detail_meta_reference_text_doi_returns_before_title_lookup() -> None:
    title_lookup_called = False

    def fetch_best_crossref_for_reference(**kwargs):
        assert kwargs["reference_text"].startswith("Demo authors")
        return {"doi": "10.1000/ref", "title": "Reference text title"}

    def fetch_best_crossref_meta(**kwargs):
        assert kwargs["doi_hint"] == "10.1000/ref"
        return {"doi": "10.1000/ref", "year": "2024"}

    def fetch_crossref_meta(*args, **kwargs):
        nonlocal title_lookup_called
        title_lookup_called = True
        return None

    out = pipeline.enrich_citation_detail_meta(
        {"raw": "Demo authors. Demo title. Journal, 2024."},
        normalize_reference_for_popup=lambda detail: dict(detail),
        normalize_doi_like=_norm_doi,
        extract_first_doi=lambda raw: "",
        build_doi_url=lambda doi: f"https://doi.org/{doi}",
        arxiv_backfill_meta_from_texts=lambda *texts: {},
        fallback_fill_reference_meta_from_raw=lambda meta: {},
        merge_meta_prefer_richer=_merge_meta,
        fetch_best_crossref_meta=fetch_best_crossref_meta,
        fetch_best_crossref_for_reference=fetch_best_crossref_for_reference,
        fetch_crossref_meta=fetch_crossref_meta,
        is_weak_meta_value=lambda key, value: False,
        should_try_openalex_arxiv_title=lambda meta, *, raw: False,
        openalex_arxiv_meta_by_title=lambda title: {},
        enrich_bibliometrics=lambda meta: dict(meta),
        ensure_summary_line=lambda meta, *, allow_crossref_abstract: dict(meta, summary_line="summary"),
    )

    assert out["doi"] == "10.1000/ref"
    assert out["title"] == "Reference text title"
    assert out["year"] == "2024"
    assert out["summary_line"] == "summary"
    assert title_lookup_called is False
