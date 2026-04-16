from kb.paper_guide_target_scope import (
    _build_paper_guide_target_scope,
    _extract_prompt_panel_letters,
    _normalize_paper_guide_target_scope,
    _paper_guide_target_scope_matches_text,
)


def test_extract_prompt_panel_letters_ignores_caption_clause_text():
    out = _extract_prompt_panel_letters(
        "For Figure 3 panel (f), what exactly does that panel correspond to? Point me to the exact supporting caption clause."
    )

    assert out == ["f"]


def test_build_paper_guide_target_scope_keeps_explicit_panel_list():
    scope = _build_paper_guide_target_scope(
        "Walk me through Figure 1 panels f and g.",
        prompt_family="figure_walkthrough",
    )

    assert scope["target_figure_num"] == 1
    assert scope["target_panel_letters"] == ["f", "g"]


def test_normalize_paper_guide_target_scope_accepts_target_figure_number_alias():
    scope = _normalize_paper_guide_target_scope(
        {
            "prompt_family": "figure_walkthrough",
            "target_figure_number": 3,
            "target_panel_letters": ["f"],
        }
    )

    assert scope["target_figure_num"] == 3
    assert scope["target_panel_letters"] == ["f"]


def test_target_scope_matches_explicit_box_metadata():
    scope = _build_paper_guide_target_scope(
        "From Box 1 only, what condition is given for reconstructing the image in the transform domain?",
        prompt_family="box_only",
    )

    assert _paper_guide_target_scope_matches_text(
        scope,
        text="It can be shown that the image in the transform domain can be reconstructed.",
        heading="Acquisition and image reconstruction strategies",
        box_number=1,
    )


def test_target_scope_matches_literal_section_heading():
    scope = _build_paper_guide_target_scope(
        "In the 'How a single-pixel camera works' section only, what trade-off do the authors describe?",
        prompt_family="strength_limits",
    )

    assert _paper_guide_target_scope_matches_text(
        scope,
        text="The detector stays simple, but the limited dynamic range becomes the trade-off.",
        heading="How a single-pixel camera works",
    )
