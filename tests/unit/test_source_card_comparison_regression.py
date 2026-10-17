from __future__ import annotations

import pytest

from api.reference_card_quality import attach_ref_card_polish_contract, ref_card_hit_quality
from api.source_relation_copy import source_relation_copy
from kb.claim_evidence_runtime import _enforce_named_source_section_citations


OBSERVATIONS = (
    "In many image restoration tasks, the expectation of the corrupted input data is the clean target "
    "that we seek to restore. Low-light photography is an example: a long, noise-free exposure is the "
    "average of short, independent, noisy exposures. The above findings indicate, that in these cases, "
    "and bar potential numerical issues, we can dispose of the clean targets entirely, as long as we "
    "are able to observe each source image twice – a task that is often significantly less costly than "
    "acquiring the clean target."
)


def test_two_observations_get_conditional_source_copy_and_pass_original_quality() -> None:
    ui = attach_ref_card_polish_contract({
        "display_name": "Noise2Noise", "render_locale": "zh",
        "primary_evidence": {"snippet": OBSERVATIONS}, "summary_line": "", "why_line": "",
    })
    assert "期望" in ui["summary_line"] and "两次" in ui["summary_line"]
    assert len(ui["why_line"]) >= 12
    assert ref_card_hit_quality({"ui_meta": ui})["ok"] is True


@pytest.mark.parametrize("source", [
    OBSERVATIONS.replace("is the clean target", "is not the clean target"),
    OBSERVATIONS.replace("we can dispose", "we cannot dispose"),
    OBSERVATIONS.replace("image twice", "image once"),
    OBSERVATIONS.replace("image twice", "image three times"),
    OBSERVATIONS.split("Low-light photography", 1)[1],
])
def test_incomplete_or_conflicting_observation_relation_cannot_generate_copy(source: str) -> None:
    assert source_relation_copy(source) is None


def test_observation_relation_cannot_come_from_title_or_wrong_locale() -> None:
    ui = attach_ref_card_polish_contract({
        "display_name": OBSERVATIONS, "render_locale": "zh",
        "primary_evidence": {"snippet": "An unrelated experiment."},
    })
    assert "两次" not in ui.get("summary_line", "")
    english = attach_ref_card_polish_contract({
        "display_name": "Noise2Noise", "render_locale": "en",
        "primary_evidence": {"snippet": OBSERVATIONS},
    })
    assert "两次" not in english.get("summary_line", "")


def _prune(answer: str):
    return _enforce_named_source_section_citations(
        answer,
        [
            {"text": "AlphaMethod uses independent noisy pairs for training.", "meta": {"source_name": "AlphaMethod"}},
            {"text": "BetaMethod uses individual noisy images.", "meta": {"source_name": "BetaMethod"}},
        ],
        prompt="Compare AlphaMethod and BetaMethod.", min_support_score=5,
    )


@pytest.mark.parametrize("prefix,separator,end", [("", ";", "."), ("边界：", "；", "。")])
def test_removing_unsupported_named_tail_closes_only_the_surviving_clause(prefix, separator, end) -> None:
    answer, _repairs, dropped = _prune(
        f"{prefix}AlphaMethod uses independent noisy pairs [1]{separator} AlphaMethod achieves 99 dB [1]."
    )
    assert dropped and "99" not in answer
    assert answer.endswith("[1]" + end)
    assert "independent noisy pairs" in answer


def test_citation_rebinding_does_not_close_an_original_semicolon() -> None:
    answer, repairs, dropped = _prune("AlphaMethod uses independent noisy pairs [2];")
    assert not dropped and repairs
    assert answer.endswith("[1];")


def test_removing_an_earlier_claim_does_not_close_an_original_semicolon() -> None:
    answer, _repairs, dropped = _prune(
        "AlphaMethod achieves 99 dB [1]; AlphaMethod uses independent noisy pairs [1];"
    )
    assert dropped and "99" not in answer
    assert answer.endswith("[1];")
