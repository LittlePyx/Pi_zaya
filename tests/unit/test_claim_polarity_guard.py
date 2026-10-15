from __future__ import annotations

import pytest

from kb.claim_polarity_guard import claim_polarity_conflicts
from kb.claim_evidence_runtime import _support_score, audit_and_repair_claim_evidence


@pytest.mark.parametrize("claim,evidence", [
    ("The model cannot filter irrelevant information.", "The model can filter out irrelevant information and remember relevant information."),
    ("The streaming memory discards all previous prompts and predictions.", "The streaming memory stores previous prompts and predictions."),
    ("The method trains the vision encoder while freezing the language model.", "The vision encoder is frozen. Only the language model is trained."),
    ("The method decreases reconstruction accuracy.", "The method increases reconstruction accuracy."),
    ("The model does not store previous prompts.", "The model stores previous prompts."),
    ("The vision encoder is not frozen.", "The vision encoder is frozen."),
    ("该模型不能过滤无关信息。", "The model filters irrelevant information."),
    ("训练视觉编码器，冻结语言模型。", "The vision encoder is frozen. The language model is trained."),
    ("该方法降低重建精度。", "The method increases reconstruction accuracy."),
])
def test_explicit_aligned_polarity_conflict_is_rejected(claim, evidence):
    assert claim_polarity_conflicts(claim, evidence) is True
    assert _support_score(claim, evidence) == 0


@pytest.mark.parametrize("claim,evidence", [
    ("The model filters irrelevant information.", "The model filters irrelevant information and remembers relevant information."),
    ("The model stores prior prompts.", "The model stores prior prompts but discards obsolete image features."),
    ("The vision encoder is frozen.", "The vision encoder is frozen while the language model is trained."),
    ("The model not only stores prior prompts but also predicts masks.", "The model stores prior prompts and predicts masks."),
    ("The method does not use attention.", "The paper describes the model architecture."),
    ("视觉编码器被冻结。", "The vision encoder is frozen."),
    ("冻结视觉编码器，训练语言模型。", "The vision encoder is frozen. The language model is trained."),
])
def test_no_conflict_is_not_a_claim_of_entailment(claim, evidence):
    assert claim_polarity_conflicts(claim, evidence) is False


def test_existing_claim_repair_does_not_rebind_an_opposite_claim_to_same_evidence():
    evidence = "The streaming memory stores previous prompts and predictions between video frames."
    result = audit_and_repair_claim_evidence(
        "The streaming memory discards all previous prompts and predictions [1].",
        [{"id": "paper:1", "text": evidence, "meta": {"source_path": "paper.md"}}],
        drop_unsupported_high_risk_claims=True,
    )
    text = result[0] if isinstance(result, tuple) else result.get("answer", "")
    assert "discards all previous" not in text
