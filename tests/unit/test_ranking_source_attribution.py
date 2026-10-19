from __future__ import annotations

from kb.claim_evidence_runtime import _enforce_named_source_section_citations


HITS = [
    {"text": "Self2Self reports improvements over BM3D, TWSC and CDnCNN in its experiments.", "meta": {"source_name": "self2self.pdf"}},
    {"text": "Noisier2Noise requires the ability to sample from a known noise distribution.", "meta": {"source_name": "noisier2noise.pdf"}},
]
PROMPT = "Do these papers establish a universal ranking?"


def test_ranking_question_keeps_both_named_sources_for_mixed_assumptions():
    answer = "Noisier2Noise requires a known noise distribution to sample from, while Self2Self reports improvements over BM3D and TWSC in its experiments [1]."
    repaired, _, _ = _enforce_named_source_section_citations(answer, HITS, prompt=PROMPT, min_support_score=5)
    assert "[2]" in repaired and "[1]" in repaired


def test_excluded_comparator_does_not_receive_a_factual_source_citation():
    answer = "Self2Self reports improvements over BM3D, TWSC and CDnCNN in its experiments, not Noisier2Noise [2][1]."
    repaired, _, _ = _enforce_named_source_section_citations(answer, HITS, prompt=PROMPT, min_support_score=5)
    assert "reports improvements" in repaired and "[1]" in repaired
    assert "[2]" not in repaired
