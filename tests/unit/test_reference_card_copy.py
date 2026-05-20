from __future__ import annotations

from api.reference_card_copy import (
    build_grounded_ref_why_line,
    finalize_ref_card_copy,
    looks_generic_ref_why_line,
    looks_templated_ref_why_line,
)


def test_generic_why_line_detector_catches_prompt_echo_template() -> None:
    assert looks_generic_ref_why_line(
        'This hit is directly relevant because it answers "Which paper in my library..."'
    )
    assert looks_generic_ref_why_line("这条命中直接回应用户查询，适合作为定位入口。")
    assert looks_generic_ref_why_line("该段落适合作为定位切口，因为属于当前命中证据的保守说明。")


def test_templated_why_line_detector_does_not_reject_specific_evidence() -> None:
    assert not looks_templated_ref_why_line(
        "The Related Work section names alternating direction method of multipliers as the reconstruction baseline."
    )
    assert not looks_generic_ref_why_line(
        "Related Work 中明确提及 Snapshot Compressive Imaging（SCI），直接回应用户查询。"
    )


def test_build_grounded_ref_why_line_personalizes_chinese_copy() -> None:
    out = build_grounded_ref_why_line(
        prefer_zh=True,
        focus_terms=["ADMM"],
        heading_path="SCINeRF / 2. Related Work",
        action="define",
    )

    assert "2. Related Work" in out
    assert "ADMM" in out
    assert "这条命中" not in out


def test_finalize_ref_card_copy_replaces_template_why_line() -> None:
    summary, why, changed = finalize_ref_card_copy(
        summary_line="Most existing methods employ ADMM for iterative optimization.",
        why_line="This hit is directly relevant because it matches the user question.",
        prefer_zh=False,
        focus_terms=["ADMM"],
        heading_path="2. Related Work",
        action="define",
    )

    assert summary.startswith("Most existing methods")
    assert changed is True
    assert "This hit is directly relevant" not in why
    assert "ADMM" in why
    assert "2. Related Work" in why


def test_grounded_ref_why_line_uses_summary_when_terms_are_missing() -> None:
    out = build_grounded_ref_why_line(
        prefer_zh=True,
        focus_terms=[],
        heading_path="SCINeRF / 1. Introduction",
        summary_line="The paper targets recovering 3D scenes from a single coded snapshot.",
    )

    assert "1. Introduction" in out
    assert "recovering 3D scenes" in out
    assert "这条命中" not in out
    assert "定位入口" not in out
