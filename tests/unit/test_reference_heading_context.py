from __future__ import annotations

from api import reference_heading_context as heading_context


def test_resolve_ref_ui_heading_context_sanitizes_and_splits_fallbacks() -> None:
    calls: list[dict] = []

    def sanitize(raw: str, **kwargs) -> str:
        calls.append({"fn": "sanitize", "raw": raw, **kwargs})
        return "2. Method / Setup"

    out = heading_context._resolve_ref_ui_heading_context(
        prompt="method",
        source_path="/kb/paper.md",
        heading_path=" raw heading ",
        sanitize_heading_path_ui=sanitize,
        top_heading=lambda heading: str(heading).split(" / ", 1)[0],
        is_non_navigational_heading_ui=lambda heading, **kwargs: str(heading) == "References",
        looks_like_doc_title_heading_ui=lambda heading, source_path: False,
        split_section_subsection=lambda heading: ("2. Method", "Setup"),
    )

    assert out == {
        "heading_path": "2. Method / Setup",
        "heading": "2. Method",
        "section_label": "2. Method",
        "subsection_label": "Setup",
    }
    assert calls == [
        {
            "fn": "sanitize",
            "raw": "raw heading",
            "prompt": "method",
            "source_path": "/kb/paper.md",
        }
    ]


def test_resolve_ref_ui_heading_context_clears_navigation_and_doc_title_labels() -> None:
    out = heading_context._resolve_ref_ui_heading_context(
        prompt="method",
        source_path="/kb/paper.md",
        heading_path="Paper Title / Abstract",
        heading_fallback="References",
        section_label="Paper Title",
        subsection_label="Details",
        sanitize_heading_path_ui=lambda raw, **kwargs: raw,
        top_heading=lambda heading: "Paper Title",
        is_non_navigational_heading_ui=lambda heading, **kwargs: str(heading) == "References",
        looks_like_doc_title_heading_ui=lambda heading, source_path: str(heading) == "Paper Title",
        split_section_subsection=lambda heading: ("Paper Title", "Abstract"),
    )

    assert out == {
        "heading_path": "Paper Title / Abstract",
        "heading": "",
        "section_label": "",
        "subsection_label": "",
    }


def test_should_allow_ref_summary_block_rescue_priority_rules() -> None:
    assert not heading_context._should_allow_ref_summary_block_rescue(
        prompt="Figure 2",
        source_path="",
        ref_pack_state="pending",
        allow_exact_locate=True,
        extract_figure_number=lambda prompt: 2,
        extract_equation_number=lambda prompt: 0,
        prompt_requires_explicit_focus_match=lambda prompt: True,
    )
    assert heading_context._should_allow_ref_summary_block_rescue(
        prompt="anything",
        source_path="/kb/paper.md",
        ref_pack_state="ready",
        allow_exact_locate=True,
        extract_figure_number=lambda prompt: 0,
        extract_equation_number=lambda prompt: 0,
        prompt_requires_explicit_focus_match=lambda prompt: False,
    )
    assert heading_context._should_allow_ref_summary_block_rescue(
        prompt="Figure 2",
        source_path="/kb/paper.md",
        ref_pack_state="ready",
        allow_exact_locate=False,
        extract_figure_number=lambda prompt: 2,
        extract_equation_number=lambda prompt: 0,
        prompt_requires_explicit_focus_match=lambda prompt: False,
    )
    assert not heading_context._should_allow_ref_summary_block_rescue(
        prompt="method",
        source_path="/kb/paper.md",
        ref_pack_state="ready",
        allow_exact_locate=False,
        extract_figure_number=lambda prompt: 0,
        extract_equation_number=lambda prompt: 0,
        prompt_requires_explicit_focus_match=lambda prompt: True,
    )
    assert heading_context._should_allow_ref_summary_block_rescue(
        prompt="specific method",
        source_path="/kb/paper.md",
        ref_pack_state="pending",
        allow_exact_locate=False,
        extract_figure_number=lambda prompt: 0,
        extract_equation_number=lambda prompt: 0,
        prompt_requires_explicit_focus_match=lambda prompt: True,
    )


def test_reference_ui_resolve_ref_ui_heading_context_uses_heading_context_module(monkeypatch) -> None:
    from api import reference_ui

    calls: list[dict] = []

    def fake_resolve(**kwargs):
        calls.append(kwargs)
        return {"heading_path": "2. Method"}

    monkeypatch.setattr(heading_context, "_resolve_ref_ui_heading_context", fake_resolve)

    assert reference_ui._resolve_ref_ui_heading_context(
        prompt="method",
        source_path="/kb/paper.md",
        heading_path="2. Method",
        heading_fallback="Method",
        section_label="Section",
        subsection_label="Subsection",
    ) == {"heading_path": "2. Method"}
    assert calls == [
        {
            "prompt": "method",
            "source_path": "/kb/paper.md",
            "heading_path": "2. Method",
            "heading_fallback": "Method",
            "section_label": "Section",
            "subsection_label": "Subsection",
            "sanitize_heading_path_ui": reference_ui._sanitize_heading_path_ui,
            "top_heading": reference_ui._top_heading,
            "is_non_navigational_heading_ui": reference_ui._is_non_navigational_heading_ui,
            "looks_like_doc_title_heading_ui": reference_ui._looks_like_doc_title_heading_ui,
            "split_section_subsection": reference_ui._split_section_subsection,
        }
    ]


def test_reference_ui_should_allow_ref_summary_block_rescue_uses_heading_context_module(monkeypatch) -> None:
    from api import reference_ui

    calls: list[dict] = []

    def fake_should_allow(**kwargs):
        calls.append(kwargs)
        return True

    monkeypatch.setattr(heading_context, "_should_allow_ref_summary_block_rescue", fake_should_allow)

    assert reference_ui._should_allow_ref_summary_block_rescue(
        prompt="Figure 2",
        source_path="/kb/paper.md",
        ref_pack_state="pending",
        allow_exact_locate=False,
    ) is True
    assert calls == [
        {
            "prompt": "Figure 2",
            "source_path": "/kb/paper.md",
            "ref_pack_state": "pending",
            "allow_exact_locate": False,
            "extract_figure_number": reference_ui.extract_figure_number,
            "extract_equation_number": reference_ui.extract_equation_number,
            "prompt_requires_explicit_focus_match": reference_ui._prompt_requires_explicit_focus_match,
        }
    ]
