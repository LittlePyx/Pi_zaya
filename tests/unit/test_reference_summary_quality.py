from __future__ import annotations

from api import reference_summary_quality as quality


def _never_fragmentary(_: str) -> bool:
    return False


def _never_why_like(_: str) -> bool:
    return False


def _not_title_echo(_: str, __: str) -> bool:
    return False


def test_is_summary_quality_ok_accepts_action_and_result_signal() -> None:
    text = (
        "We propose an adaptive sampling method for single-pixel imaging that selects "
        "informative illumination patterns during acquisition. Experiments show improved "
        "reconstruction quality under limited measurements."
    )

    assert quality._is_summary_quality_ok(
        text,
        looks_fragmentary_ref_summary=_never_fragmentary,
        looks_why_like_ref_summary=_never_why_like,
    )


def test_is_summary_quality_ok_rejects_fragmentary_summary() -> None:
    text = (
        "We propose an adaptive sampling method for single-pixel imaging. Experiments show "
        "improved reconstruction quality under limited measurements."
    )

    assert not quality._is_summary_quality_ok(
        text,
        looks_fragmentary_ref_summary=lambda _: True,
        looks_why_like_ref_summary=_never_why_like,
    )


def test_summary_quality_contract_marks_grounded_abstract_ready() -> None:
    contract = quality._summary_quality_contract(
        {
            "title": "Adaptive sampling for single-pixel imaging",
            "summary_line": (
                "We propose an adaptive sampling method for single-pixel imaging that selects "
                "informative illumination patterns. Experiments show improved reconstruction "
                "quality under limited measurements."
            ),
            "summary_source": "abstract",
            "summary_provider": "crossref",
            "summary_generation": "translated_abstract",
        },
        is_summary_quality_ok=lambda _: True,
        looks_like_title_echo=_not_title_echo,
    )

    assert contract["status"] == "grounded"
    assert contract["ok"] is True
    assert contract["export_ready"] is True
    assert contract["score"] >= 92


def test_summary_quality_contract_marks_metadata_fallback_not_export_ready() -> None:
    contract = quality._summary_quality_contract(
        {
            "summary_line": "No abstract is available for this paper yet, so this card falls back to metadata only.",
            "summary_source": "metadata",
            "summary_generation": "metadata_only",
        },
        is_summary_quality_ok=lambda _: False,
        looks_like_title_echo=_not_title_echo,
    )

    assert contract["status"] == "fallback"
    assert contract["export_ready"] is False
    assert {issue["code"] for issue in contract["issues"]} >= {"metadata_only_summary", "metadata_only"}


def test_summary_quality_contract_flags_title_echo() -> None:
    contract = quality._summary_quality_contract(
        {
            "title": "Adaptive sampling for single-pixel imaging",
            "summary_line": "Adaptive sampling for single-pixel imaging",
            "summary_source": "abstract",
        },
        is_summary_quality_ok=lambda _: True,
        looks_like_title_echo=lambda _summary, _title: True,
    )

    assert contract["ok"] is False
    assert any(issue["code"] == "title_echo" for issue in contract["issues"])


def test_low_value_and_metadata_only_patterns() -> None:
    assert quality._looks_low_value_shelf_summary(
        "This cited prior work helps verify where the method background comes from."
    )
    assert quality._looks_metadata_only_summary(
        "Only limited bibliographic metadata is currently available, and no abstract was found."
    )
