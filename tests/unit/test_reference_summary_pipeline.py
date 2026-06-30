from __future__ import annotations

from api import reference_summary_pipeline as pipeline


def _attach(meta: dict) -> dict:
    out = dict(meta)
    out["summary_quality"] = {"ok": True}
    return out


def _finalize(*, title: str, abstract_text: str) -> tuple[str, str]:
    return f"Finalized: {abstract_text}", "llm_abstract"


def _run_pipeline(meta: dict, **overrides) -> dict:
    kwargs = {
        "allow_crossref_abstract": True,
        "looks_low_value_shelf_summary": lambda text: False,
        "looks_like_title_echo": lambda summary, title: False,
        "looks_metadata_only_summary": lambda text: False,
        "finalize_abstract_summary_line": _finalize,
        "translate_summary_to_zh": lambda text: f"Translated: {text}",
        "attach_summary_quality": _attach,
        "summary_from_crossref_abstract": lambda meta: "",
        "summary_from_openalex_abstract": lambda meta: "",
        "summary_from_semantic_scholar_abstract": lambda meta: "",
        "summary_from_doi_landing_page": lambda meta: "",
        "contextual_summary_line": lambda meta: "",
        "metadata_summary_line": lambda meta: "Metadata fallback summary without usable abstract text.",
    }
    kwargs.update(overrides)
    return pipeline._ensure_summary_line(meta, **kwargs)


def test_existing_abstract_summary_is_finalized_and_marked_abstract() -> None:
    out = _run_pipeline(
        {
            "title": "Adaptive sampling for single-pixel imaging",
            "summary_line": "We propose an adaptive sampling method and experiments show improved reconstruction quality.",
            "summary_source": "abstract",
        }
    )

    assert out["summary_line"].startswith("Finalized: We propose an adaptive sampling method")
    assert out["summary_source"] == "abstract"
    assert out["summary_generation"] == "llm_abstract"
    assert out["summary_quality"]["ok"] is True


def test_existing_non_abstract_summary_is_translated_and_normalized_to_fulltext() -> None:
    out = _run_pipeline(
        {
            "summary_line": "This section describes adaptive sampling and reports improved reconstruction results.",
            "summary_source": "navigation",
        }
    )

    assert out["summary_line"].startswith("Translated: This section describes")
    assert out["summary_source"] == "fulltext"
    assert out["summary_generation"] == "fulltext_existing"


def test_external_abstract_providers_are_tried_in_order() -> None:
    calls: list[str] = []

    def empty(name: str):
        def _inner(meta: dict) -> str:
            calls.append(name)
            return ""

        return _inner

    def semantic(meta: dict) -> str:
        calls.append("semantic")
        return "We develop an adaptive sampling method and experiments show improved reconstruction quality."

    out = _run_pipeline(
        {"title": "Adaptive sampling for single-pixel imaging"},
        summary_from_crossref_abstract=empty("crossref"),
        summary_from_openalex_abstract=empty("openalex"),
        summary_from_semantic_scholar_abstract=semantic,
        summary_from_doi_landing_page=empty("landing"),
    )

    assert calls == ["crossref", "openalex", "semantic"]
    assert out["summary_provider"] == "semantic_scholar"
    assert out["summary_source"] == "abstract"
    assert out["summary_generation"] == "llm_abstract"


def test_contextual_fallback_precedes_metadata_when_external_lookup_disabled() -> None:
    out = _run_pipeline(
        {"title": "No external lookup"},
        allow_crossref_abstract=False,
        contextual_summary_line=lambda meta: "Citation context explains how this work supports adaptive sampling.",
        metadata_summary_line=lambda meta: "Metadata fallback should not be used.",
    )

    assert out["summary_line"] == "Citation context explains how this work supports adaptive sampling."
    assert out["summary_source"] == "citation_context"
    assert out["summary_generation"] == "citation_context_fallback"


def test_metadata_fallback_is_used_when_all_sources_are_empty() -> None:
    out = _run_pipeline(
        {"title": "No summary available"},
        allow_crossref_abstract=False,
        contextual_summary_line=lambda meta: "",
        metadata_summary_line=lambda meta: "Metadata fallback summary without usable abstract text.",
    )

    assert out["summary_line"] == "Metadata fallback summary without usable abstract text."
    assert out["summary_source"] == "metadata"
    assert out["summary_generation"] == "metadata_only"


def test_low_value_existing_summary_is_discarded_before_fallback() -> None:
    out = _run_pipeline(
        {
            "summary_line": "This cited prior work helps verify where the method background comes from.",
            "summary_source": "fulltext",
        },
        allow_crossref_abstract=False,
        looks_low_value_shelf_summary=lambda text: True,
        contextual_summary_line=lambda meta: "",
        metadata_summary_line=lambda meta: "Metadata fallback summary without usable abstract text.",
    )

    assert out["summary_line"] == "Metadata fallback summary without usable abstract text."
    assert out["summary_source"] == "metadata"
    assert out["summary_generation"] == "metadata_only"
