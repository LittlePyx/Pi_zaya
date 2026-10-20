from __future__ import annotations

import pytest

from kb.direct_question_evidence import attach_direct_source_evidence, direct_question_evidence_hits
from kb.paper_guide_answer_selection import _merge_same_source_answer_hits


QUESTION = "比较两篇的相关噪声条件，区分图内分量与额外生成噪声的独立性关系。"
SPATIAL = "We also allow spatially correlated noise models. On the other hand, we require the ability to sample from the noise distribution."
SAME = "In our construction the original noisy image receives a synthetic sample from the same noise distribution."
IID = "Given the sum of two i.i.d. noise samples, the best estimate of one of them is half of their sum."


def test_two_passage_budget_retains_sample_independence_and_spatial_condition(tmp_path):
    source = tmp_path / "method.md"
    source.write_text(f"<!-- kb_page: 2 -->\n## Related Work\n\n{SPATIAL}\n\n<!-- kb_page: 3 -->\n## Method\n\n{SAME}\n\n{IID}\n", encoding="utf8")
    hits = direct_question_evidence_hits(source, QUESTION, limit=2)
    assert len(hits) == 2
    assert any(h["text"] == IID for h in hits)
    assert any("spatially correlated" in h["text"] for h in hits)
    merged = _merge_same_source_answer_hits(hits)
    slot = {**hits[0]["meta"], "preferred_system": "system_a", "evidence_quote": hits[0]["text"]}
    attached = attach_direct_source_evidence({"slots": [slot]}, merged)["slots"][0]
    assert {r["page_start"] for r in attached["direct_source_evidence"]} == {2, 3}
    assert IID in attached["source_evidence_quotes"]
    from api.source_relation_copy import source_relation_copy
    copy = source_relation_copy(IID)
    assert copy and "两份" in copy[0] and "内部" in copy[1]


@pytest.mark.parametrize("relation", [
    IID.replace("i.i.d.", "spatially correlated"),
    "Two noise samples have the same distribution, but are not independent.",
    "Two independent noise samples can have different distributions.",
    "It is false that " + IID,
    "Within each image the noise components are independent and identically distributed.",
])
def test_other_noise_relations_do_not_displace_spatial_premise(tmp_path, relation):
    source = tmp_path / "method.md"
    source.write_text(f"<!-- kb_page: 2 -->\n{SPATIAL}\n\n{SAME}\n\n{relation}", encoding="utf8")
    hits = direct_question_evidence_hits(source, QUESTION, limit=2)
    assert not any(h["text"] == relation for h in hits)
    assert any("spatially correlated" in h["text"] for h in hits)


def test_iid_abbreviation_is_not_a_claim_boundary():
    from kb.claim_evidence_runtime import _split_claim_segments
    sentence = "该估计使用两个 i.i.d. 噪声样本，而不要求每个样本内部各分量独立 [2]。"
    assert _split_claim_segments(sentence) == [sentence]


@pytest.mark.parametrize("named", [False, True])
def test_spatial_sampling_claim_selects_its_own_retrieved_passage(tmp_path, named):
    from api.claim_passage_plan import message_citation_plan, prepare_claim_passage_plan
    source = tmp_path / "noisier2noise.md"
    spatial = SPATIAL.replace("noise models.", "noise models, which are problematic for Noise2Void and Noise2Self.")
    source.write_text(f"<!-- kb_page: 2 -->\n## Method\n\n{IID}\n\n{spatial}", encoding="utf8")
    hits = direct_question_evidence_hits(source, QUESTION, limit=2)
    slot = {**hits[0]["meta"], "source_name": "noisier2noise", "preferred_system": "system_a", "candidate_hits": [1], "evidence_quote": hits[0]["text"]}
    plan = attach_direct_source_evidence({"slots": [slot]}, _merge_same_source_answer_hits(hits))
    claim = ("Noisier2Noise" if named else "该方法") + "允许空间相关噪声模型，但要求能够从噪声分布中采样 [1]。"
    rec = {"meta": {"canonical_hit_paths": [str(source)], "answer_quality": {"citation_plan": plan}}}
    result = prepare_claim_passage_plan(rec, {"hits": _merge_same_source_answer_hits(hits)}, claim)
    slots = message_citation_plan(result)["slots"]
    assert len(slots) == 2
    assert "spatially correlated" in slots[1]["evidence_quote"]
    assert slots[1]["block_id"] != slots[0]["block_id"]


def test_spatial_and_sampling_claims_cannot_borrow_iid_only_evidence():
    from api.claim_passage_plan import _supports
    assert not _supports("This method allows spatially correlated noise [1].", IID, "Method", "method")
    assert not _supports("This method requires sampling from the noise distribution [1].", IID, "Method", "method")


@pytest.mark.parametrize("evidence", [
    SPATIAL.replace("We also allow", "We do not allow"),
    SPATIAL.replace("we require", "we do not require"),
])
def test_bilingual_relation_alignment_keeps_opposite_premises_rejected(evidence):
    from api.claim_passage_plan import _supports
    claim = "该方法允许空间相关噪声模型，但要求能够从噪声分布中采样 [1]。"
    assert not _supports(claim, evidence, "", "method")


def test_training_comparison_keeps_bernoulli_observation_evidence(tmp_path):
    from kb.direct_question_evidence import prioritize_selected_source_relation_hits
    selected = [
        {"text": "We train on Bernoulli-masked instances of the single noisy image.", "meta": {"source_path": str(tmp_path / "self2self.md")}},
        {"text": "We require a statistical noise model and a single noisy observation.", "meta": {"source_path": str(tmp_path / "noisier2noise.md")}},
    ]
    source = tmp_path / "noisier2noise.md"
    source.write_text('<!-- kb_page: 1 -->\nOur method requires a known noise model.\n\nWe require the ability to sample from the noise distribution.', encoding='utf8')
    result = prioritize_selected_source_relation_hits(selected, [], limit=2, prompt='Compare the training observations and noise model required by Self2Self and Noisier2Noise.')
    assert result == selected
    assert 'Bernoulli' in result[0]['text']


def test_additive_training_pair_keeps_direction_and_synthetic_noise_premise(tmp_path):
    source = tmp_path / 'additive.md'
    premise = 'Because the noise distribution is known, we can draw an additional synthetic noise sample.'
    direction = 'train the network to predict Q given P as input, using a pixel-wise L2 loss function and standard stochastic gradient descent.'
    source.write_text(f'<!-- kb_page: 3 -->\n## Method\n\n{premise}\n\n{direction}', encoding='utf8')
    hits = direct_question_evidence_hits(source, 'For the standard additive procedure, distinguish the network input P from its training target Q.')
    assert {h['text'] for h in hits} == {premise, direction}
    assert all(h['meta']['page_start'] == 3 for h in hits)
