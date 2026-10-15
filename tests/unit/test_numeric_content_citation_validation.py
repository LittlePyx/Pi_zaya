from __future__ import annotations


def test_source_facet_completion_preserves_noise_interval_and_removes_reference_decoration():
    from kb.generation_answer_finalize_runtime import _complete_grounded_requested_source_facets
    evidence = r"During training, noisy Gaussian images cover noise levels $\sigma\in[7,47]$ without requiring noise-level estimation [12]."
    hits = [{"text": evidence, "meta": {"source_path": "paper.md"}}]
    plan = {"slots": [{"preferred_system": "system_a", "source_path": "paper.md", "candidate_hits": [1], "evidence_quote": evidence}]}
    completed = _complete_grounded_requested_source_facets("One model is trained.", prompt="What exact training noise-level interval is used for Gaussian denoising?", citation_plan=plan, answer_hits=hits)
    assert "[7,47]" in completed
    assert "[12]" not in completed
    assert "[1]" in completed


def test_claim_normalization_does_not_delete_numeric_noise_intervals():
    from kb.claim_evidence_runtime import _plain_claim
    claim = r"The noise range is $\sigma\in[0,55]$ [1]."
    cleaned = _plain_claim(claim)
    assert "[0,55]" in cleaned and "[1]" not in cleaned

import pytest

from kb.generation_citation_validation_runtime import _validate_freeform_numeric_citations


@pytest.mark.parametrize("value", [
    r"$\sigma \in [0,55]$", r"$$x \in [1,9]$$", r"\(x \in [1,9]\)",
    r"\[x \in [1,9]\]", "range of [5,55]", "`array[9]`", "```python\na[9]\n```",
])
def test_numeric_content_survives_and_invalid_prose_citation_is_removed(value):
    answer = value + " Supported [1]. Invalid [99]."
    cleaned, stats = _validate_freeform_numeric_citations(answer, answer_hits=[{"text": "Supported."}])
    assert value in cleaned
    assert "Supported [1]" in cleaned
    assert "[99]" not in cleaned
    assert stats["raw_count"] == 2 and stats["dropped"] == 1
