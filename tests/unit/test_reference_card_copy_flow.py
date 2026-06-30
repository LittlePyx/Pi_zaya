from __future__ import annotations

from api import reference_card_copy_flow as copy_flow


def test_resolve_ref_card_why_line_replaces_navigation_with_prompt_aligned_focus() -> None:
    out = copy_flow._resolve_ref_card_why_line(
        prompt="What defines ADMM?",
        display_name="Paper.pdf",
        heading_path="2. Related Work",
        heading="Related Work",
        section_label="Related Work",
        subsection_label="",
        nav={"why": "General navigation note.", "find": []},
        summary_line="The section discusses optimization.",
        fallback_why_line_ui=lambda **kwargs: "fallback note",
        build_prompt_aligned_ref_why_line=lambda **kwargs: "This section defines ADMM.",
        matched_focus_terms_for_ref_card=lambda prompt, **kwargs: (
            ["ADMM"] if "ADMM" in str(kwargs.get("surface_text") or "") else []
        ),
        is_definition_focus_prompt=lambda prompt: True,
        why_line_explicitly_names_focus_term=lambda prompt, why: "ADMM" in str(why),
    )

    assert out == {
        "why_line": "This section defines ADMM.",
        "why_generation": "deterministic_grounded",
    }


def test_resolve_ref_card_why_line_uses_fallback_when_navigation_is_empty() -> None:
    fallback_calls: list[dict] = []

    def fallback(**kwargs) -> str:
        fallback_calls.append(kwargs)
        return "Fallback grounded note."

    out = copy_flow._resolve_ref_card_why_line(
        prompt="method",
        display_name="Paper.pdf",
        heading_path="",
        heading="3. Method",
        section_label="Method",
        subsection_label="Setup",
        nav={"find": ["mask"]},
        summary_line="The method uses coded masks.",
        fallback_why_line_ui=fallback,
        build_prompt_aligned_ref_why_line=lambda **kwargs: "",
        matched_focus_terms_for_ref_card=lambda prompt, **kwargs: [],
        is_definition_focus_prompt=lambda prompt: False,
        why_line_explicitly_names_focus_term=lambda prompt, why: False,
    )

    assert out["why_line"] == "Fallback grounded note."
    assert out["why_generation"] == "deterministic_grounded"
    assert fallback_calls == [
        {
            "prompt": "method",
            "heading_label": "3. Method",
            "section_label": "Method",
            "subsection_label": "Setup",
            "find_terms": ["mask"],
        }
    ]


def test_resolve_ref_card_summary_kind_and_copy_flips_generation_when_finalize_changes() -> None:
    finalize_calls: list[dict] = []

    def finalize(**kwargs):
        finalize_calls.append(kwargs)
        return "Final summary.", "Final why.", True

    out = copy_flow._resolve_ref_card_summary_kind_and_copy(
        prompt="Define ADMM",
        display_name="Paper.pdf",
        heading_path="2. Related Work",
        heading="Related Work",
        summary_line="Raw summary.",
        why_line="Template why.",
        why_generation="navigation",
        citation_meta={"summary_line": "Raw summary."},
        used_prompt_aligned_summary=False,
        used_nav_summary=True,
        allow_llm_translate=False,
        infer_ref_summary_kind=lambda **kwargs: "guide",
        align_ref_card_copy_to_user_locale=lambda **kwargs: ("Aligned summary.", "Aligned why."),
        matched_focus_terms_for_ref_card=lambda prompt, **kwargs: ["admm"],
        display_focus_term_for_ref_card=lambda prompt, term: str(term).upper(),
        ref_card_user_locale=lambda *args: "en",
        finalize_ref_card_copy=finalize,
        prompt_reference_focus_action=lambda prompt: "define",
    )

    assert out["summary_line"] == "Final summary."
    assert out["why_line"] == "Final why."
    assert out["why_generation"] == "deterministic_grounded"
    assert out["summary_kind"] == "guide"
    assert out["render_locale"] == "en"
    assert finalize_calls[0]["focus_terms"] == ["ADMM"]
    assert finalize_calls[0]["action"] == "define"


def test_build_ref_card_basis_bundle_uses_abstract_generation_from_citation_meta() -> None:
    out = copy_flow._build_ref_card_basis_bundle(
        prompt="summarize",
        citation_meta={"summary_generation": "llm_abstract"},
        summary_kind="abstract",
        summary_line="Abstract summary.",
        why_generation="deterministic_grounded",
        why_line="Why.",
        build_ref_summary_surface_meta=lambda **kwargs: {
            "summary_kind": kwargs["summary_kind"],
            "summary_label": "Abstract",
        },
        build_ref_summary_basis_meta=lambda **kwargs: {
            "summary_generation": kwargs["summary_generation"],
            "summary_basis": "summary basis",
        },
        build_ref_why_basis_meta=lambda **kwargs: {
            "why_generation": kwargs["why_generation"],
            "why_basis": "why basis",
        },
    )

    assert out["summary_generation"] == "llm_abstract"
    assert out["summary_surface"] == {"summary_kind": "abstract", "summary_label": "Abstract"}
    assert out["summary_basis_meta"] == {
        "summary_generation": "llm_abstract",
        "summary_basis": "summary basis",
    }
    assert out["why_basis_meta"] == {
        "why_generation": "deterministic_grounded",
        "why_basis": "why basis",
    }


def test_reference_ui_ref_card_copy_flow_proxies_use_module_dependencies(monkeypatch) -> None:
    from api import reference_ui

    calls: list[dict] = []

    def fake_why(**kwargs):
        calls.append(kwargs)
        return {"why_line": "why", "why_generation": "fallback"}

    monkeypatch.setattr(copy_flow, "_resolve_ref_card_why_line", fake_why)

    out = reference_ui._resolve_ref_card_why_line(
        prompt="method",
        display_name="Paper.pdf",
        heading_path="2. Method",
        heading="Method",
        section_label="Section",
        subsection_label="Subsection",
        nav={"find": []},
        summary_line="Summary.",
    )

    assert out == {"why_line": "why", "why_generation": "fallback"}
    assert calls
    call = calls[0]
    assert call["fallback_why_line_ui"] is reference_ui._fallback_why_line_ui
    assert call["build_prompt_aligned_ref_why_line"] is reference_ui._build_prompt_aligned_ref_why_line_v3
    assert call["matched_focus_terms_for_ref_card"] is reference_ui._matched_focus_terms_for_ref_card
    assert call["is_definition_focus_prompt"] is reference_ui._is_definition_focus_prompt
    assert call["why_line_explicitly_names_focus_term"] is reference_ui._why_line_explicitly_names_focus_term


def test_reference_ui_ref_card_summary_copy_proxy_uses_module_dependencies(monkeypatch) -> None:
    from api import reference_ui

    calls: list[dict] = []

    def fake_copy(**kwargs):
        calls.append(kwargs)
        return {
            "summary_line": "summary",
            "why_line": "why",
            "why_generation": "deterministic_grounded",
            "summary_kind": "guide",
            "render_locale": "en",
        }

    monkeypatch.setattr(copy_flow, "_resolve_ref_card_summary_kind_and_copy", fake_copy)

    out = reference_ui._resolve_ref_card_summary_kind_and_copy(
        prompt="method",
        display_name="Paper.pdf",
        heading_path="2. Method",
        heading="Method",
        summary_line="Summary.",
        why_line="Why.",
        why_generation="navigation",
        citation_meta={},
        used_prompt_aligned_summary=False,
        used_nav_summary=True,
        allow_llm_translate=False,
    )

    assert out["render_locale"] == "en"
    assert calls
    call = calls[0]
    assert call["infer_ref_summary_kind"] is reference_ui._infer_ref_summary_kind
    assert call["align_ref_card_copy_to_user_locale"] is reference_ui._align_ref_card_copy_to_user_locale
    assert call["matched_focus_terms_for_ref_card"] is reference_ui._matched_focus_terms_for_ref_card
    assert call["display_focus_term_for_ref_card"] is reference_ui._display_focus_term_for_ref_card
    assert call["ref_card_user_locale"] is reference_ui._ref_card_user_locale
    assert call["finalize_ref_card_copy"] is reference_ui._finalize_ref_card_copy
    assert call["prompt_reference_focus_action"] is reference_ui._shared_prompt_reference_focus_action


def test_reference_ui_ref_card_basis_bundle_proxy_uses_module_dependencies(monkeypatch) -> None:
    from api import reference_ui

    calls: list[dict] = []

    def fake_basis(**kwargs):
        calls.append(kwargs)
        return {
            "summary_surface": {},
            "summary_generation": "section_grounded",
            "summary_basis_meta": {},
            "why_basis_meta": {},
        }

    monkeypatch.setattr(copy_flow, "_build_ref_card_basis_bundle", fake_basis)

    out = reference_ui._build_ref_card_basis_bundle(
        prompt="method",
        citation_meta={},
        summary_kind="guide",
        summary_line="Summary.",
        why_generation="fallback",
        why_line="Why.",
    )

    assert out["summary_generation"] == "section_grounded"
    assert calls
    call = calls[0]
    assert call["build_ref_summary_surface_meta"] is reference_ui._build_ref_summary_surface_meta
    assert call["build_ref_summary_basis_meta"] is reference_ui._build_ref_summary_basis_meta
    assert call["build_ref_why_basis_meta"] is reference_ui._build_ref_why_basis_meta
