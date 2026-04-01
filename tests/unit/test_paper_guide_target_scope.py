from kb.paper_guide_target_scope import (
    _build_paper_guide_target_scope,
    _extract_prompt_panel_letters,
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
