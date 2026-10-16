from __future__ import annotations

import pytest

from kb.claim_evidence_runtime import _support_score, audit_and_repair_claim_evidence
from kb.evidence_binding import assess_system_a_hit_binding


SOURCE = (
    "While it cannot be expected that our approach outperforms methods that have "
    "additional information available during training, we observe that the denoising "
    "performance of our results only drops moderately and is still outperforming BM3D."
)
CLAIM = "原文不支持这种训练方式应当胜过拥有更多训练信息的方法"


def _hit(text):
    return {"text": text, "meta": {
        "citation_plan_evidence_authoritative": True,
        "citation_plan_full_evidence_quote": text,
        "citation_plan_evidence_quotes": [text],
        "citation_plan_evidence_selection_reason": "prompt_aligned_source_sentence",
    }}


def _binding(claim, evidence):
    hit = _hit(evidence)
    return assess_system_a_hit_binding(answer_claim=claim, hit=hit, meta=hit["meta"],
        heading="", evidence_quote=evidence, source_name="")


def test_chinese_information_boundary_survives_both_gates():
    assert _support_score(CLAIM, SOURCE) >= 5
    assert not _binding(CLAIM, SOURCE)["suppress_link"]
    answer, _ = audit_and_repair_claim_evidence(CLAIM + " [1]。", [_hit(SOURCE)],
        allowed_citation_numbers={1}, drop_unsupported_unplanned_claims=True,
        drop_unsupported_high_risk_claims=True, enforce_user_visible_binding=True)
    assert CLAIM in answer and "[1]" in answer


@pytest.mark.parametrize("claim", [
    CLAIM.replace("不支持", "支持"),
    CLAIM.replace("不支持这种训练方式应当", "表明这种训练方式绝对不能"),
    CLAIM.replace("不支持这种训练方式应当", "表明这种训练方式永远不会"),
    CLAIM.replace("更多", "更少"),
    CLAIM.replace("更多", "相同"),
    "原文不支持拥有更多训练信息的方法应当胜过这种训练方式。",
    "拥有更多训练信息的方法优于这种训练方式。",
    "Our approach can be expected to outperform methods with more training information.",
    "Our approach never outperforms methods with additional information available during training.",
])
def test_changed_comparator_or_modal_strength_is_rejected(claim):
    assert _support_score(claim, SOURCE) == 0
    assert _binding(claim, SOURCE)["suppress_link"]


@pytest.mark.parametrize("evidence", [
    SOURCE.replace("cannot be expected", "can be expected"),
    SOURCE.replace("additional information", "less information"),
    SOURCE.replace("additional information", "the same information"),
    "Training uses additional information. Our approach outperforms BM3D.",
])
def test_source_must_state_the_comparative_boundary(evidence):
    assert _support_score(CLAIM, evidence) == 0
    assert _binding(CLAIM, evidence)["suppress_link"]
