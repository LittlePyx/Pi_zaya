from __future__ import annotations

from api import reference_detail_arxiv as arxiv


def _merge(base: dict, incoming: dict) -> dict:
    return {**dict(base or {}), **dict(incoming or {})}


def test_merge_existing_arxiv_backfill_uses_existing_fields_when_no_doi() -> None:
    seen: list[tuple[str, ...]] = []

    def backfill(*texts: str) -> dict:
        seen.append(tuple(texts))
        return {"doi": "10.48550/arXiv.2008.03824", "doi_url": "https://doi.org/10.48550/arXiv.2008.03824"}

    out = arxiv.merge_existing_arxiv_backfill(
        {"raw": "arXiv:2008.03824", "title": "Neural reflectance fields", "venue": "arXiv"},
        arxiv_backfill_meta_from_texts=backfill,
        normalize_doi_like=lambda value: "",
        merge_meta_prefer_richer=_merge,
    )

    assert out["doi"] == "10.48550/arXiv.2008.03824"
    assert seen[0][2] == "arXiv:2008.03824"


def test_merge_existing_arxiv_backfill_does_not_merge_when_doi_present() -> None:
    out = arxiv.merge_existing_arxiv_backfill(
        {"doi": "10.1000/existing", "raw": "arXiv:2008.03824"},
        arxiv_backfill_meta_from_texts=lambda *texts: {"doi": "10.48550/arXiv.2008.03824"},
        normalize_doi_like=lambda value: "10.1000/existing",
        merge_meta_prefer_richer=_merge,
    )

    assert out["doi"] == "10.1000/existing"


def test_merge_missing_doi_arxiv_fallback_prefers_text_backfill_before_openalex() -> None:
    openalex_calls: list[str] = []

    out = arxiv.merge_missing_doi_arxiv_fallback(
        {"title": "Neural reflectance fields", "venue": "arXiv preprint"},
        raw_seed="arXiv:2008.03824",
        raw="",
        title="Neural reflectance fields",
        venue="arXiv preprint",
        arxiv_backfill_meta_from_texts=lambda *texts: {"doi": "10.48550/arXiv.2008.03824"},
        normalize_doi_like=lambda value: "10.48550/arxiv.2008.03824" if value else "",
        merge_meta_prefer_richer=_merge,
        should_try_openalex_arxiv_title=lambda meta, *, raw: True,
        openalex_arxiv_meta_by_title=lambda title: openalex_calls.append(title) or {"doi": "wrong"},
    )

    assert out["doi"] == "10.48550/arXiv.2008.03824"
    assert openalex_calls == []


def test_merge_missing_doi_arxiv_fallback_uses_openalex_when_text_backfill_empty() -> None:
    out = arxiv.merge_missing_doi_arxiv_fallback(
        {"title": "Neural reflectance fields", "venue": "arXiv preprint"},
        raw_seed="",
        raw="arXiv preprint, 2020",
        title="Neural reflectance fields",
        venue="arXiv preprint",
        arxiv_backfill_meta_from_texts=lambda *texts: {},
        normalize_doi_like=lambda value: "",
        merge_meta_prefer_richer=_merge,
        should_try_openalex_arxiv_title=lambda meta, *, raw: "arXiv" in raw or "arxiv" in str(meta.get("venue") or "").lower(),
        openalex_arxiv_meta_by_title=lambda title: {
            "doi": "10.48550/arXiv.2008.03824",
            "doi_url": "https://doi.org/10.48550/arXiv.2008.03824",
        },
    )

    assert out["doi"] == "10.48550/arXiv.2008.03824"
    assert out["doi_url"] == "https://doi.org/10.48550/arXiv.2008.03824"


def test_merge_missing_doi_arxiv_fallback_noops_when_doi_already_present() -> None:
    out = arxiv.merge_missing_doi_arxiv_fallback(
        {"doi": "10.1000/existing"},
        raw_seed="arXiv:2008.03824",
        raw="",
        title="Neural reflectance fields",
        venue="arXiv",
        arxiv_backfill_meta_from_texts=lambda *texts: {"doi": "wrong"},
        normalize_doi_like=lambda value: str(value or ""),
        merge_meta_prefer_richer=_merge,
        should_try_openalex_arxiv_title=lambda meta, *, raw: True,
        openalex_arxiv_meta_by_title=lambda title: {"doi": "wrong"},
    )

    assert out == {"doi": "10.1000/existing"}
