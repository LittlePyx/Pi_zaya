from __future__ import annotations

import pytest

from kb.generation_answer_finalize_runtime import _finalize_generation_answer


def _finalize(prompt: str, answer: str) -> dict:
    return _finalize_generation_answer(
        answer,
        prompt=prompt,
        prompt_for_user=prompt,
        answer_hits=[{"text": "This paper evaluates image denoising on benchmark images.",
                      "meta": {"source_path": "one.md", "source_name": "One paper"}}],
        db_dir=None,
        locked_citation_source=None,
        answer_intent="reading",
        answer_depth="L2",
        answer_output_mode="reading_guide",
        paper_guide_mode=False,
        paper_guide_contract_enabled=False,
        paper_guide_prompt_family="overview",
        paper_guide_special_focus_block="",
        paper_guide_focus_source_path="",
        paper_guide_direct_source_path="",
        paper_guide_bound_source_path="",
        paper_guide_candidate_refs_by_source={},
        paper_guide_support_slots=[],
        paper_guide_evidence_cards=[],
        apply_paper_guide_answer_postprocess=lambda text, **_: (text, []),
        maybe_append_library_figure_markdown=lambda text, **_: text,
        validate_structured_citations=lambda text, **_: (text, {}),
    )


@pytest.mark.parametrize("prompt", [
    "仅依据这两篇论文，给出同一项随机临床试验的患者人数。如果没有证据，不要假设数字。",
    "Do these two papers establish a universal ranking? State the evidence boundary instead of inventing a result.",
    "Compare these two papers' assumptions, using each paper's own evidence.",
])
def test_fixed_source_question_does_not_require_numbered_paper_summaries(prompt: str) -> None:
    answer = (
        "The selected excerpts do not establish the requested comparison. "
        "This is a limit of the retrieved window, not proof about the full papers."
    )
    result = _finalize(prompt, answer)

    assert result["answer_quality"]["minimum_ok"] is True
    assert "paper_count_ok" not in result["answer_quality"]


def test_explicit_paper_selection_still_rejects_a_missing_requested_item() -> None:
    result = _finalize("List exactly 2 papers on image denoising.",
                       "1. One paper evaluates image denoising on benchmark images [1].")

    assert result["answer_quality"]["requested_paper_count"] == 2
    assert result["answer_quality"]["paper_count_ok"] is False
    assert result["answer_quality"]["minimum_ok"] is False
