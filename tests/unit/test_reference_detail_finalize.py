from __future__ import annotations

from api import reference_detail_finalize as finalize


def test_enrich_bibliometrics_and_summary_applies_bibliometrics_then_summary() -> None:
    seen: list[dict] = []

    def enrich(meta: dict) -> dict:
        return {**meta, "citation_count": 42}

    def ensure(meta: dict, *, allow_crossref_abstract: bool) -> dict:
        seen.append({"meta": dict(meta), "allow": allow_crossref_abstract})
        return {**meta, "summary_line": "Summary"}

    out = finalize.enrich_bibliometrics_and_summary(
        {"title": "Demo"},
        enrich_bibliometrics=enrich,
        ensure_summary_line=ensure,
        allow_crossref_abstract=True,
    )

    assert out["citation_count"] == 42
    assert out["summary_line"] == "Summary"
    assert seen == [{"meta": {"title": "Demo", "citation_count": 42}, "allow": True}]


def test_enrich_bibliometrics_and_summary_ignores_non_dict_enrichment() -> None:
    out = finalize.enrich_bibliometrics_and_summary(
        {"title": "Demo"},
        enrich_bibliometrics=lambda meta: None,  # type: ignore[return-value]
        ensure_summary_line=lambda meta, **kwargs: {**meta, "summary_line": "Summary"},
        allow_crossref_abstract=False,
    )

    assert out == {"title": "Demo", "summary_line": "Summary"}
