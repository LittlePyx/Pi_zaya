from __future__ import annotations

from pathlib import Path

import pytest

from kb.claim_evidence_runtime import _hit_payload, _support_score, audit_and_repair_claim_evidence
from kb.direct_question_evidence import attach_direct_source_evidence
from kb.evidence_binding import explicit_claim_relations_covered
from kb.generation_answer_finalize_runtime import _claim_evidence_hits_with_citation_plan
from kb.source_blocks import load_source_blocks


BODY = (
    "A trivial property of L 2 minimization is that on expectation, the estimate remains unchanged "
    "if we replace the targets with random numbers whose expectations match the targets. "
    "Practical experiments in Sections 1, 3 and 4 study various cases where clean targets are not needed. "
    "The optimal network parameters remain unchanged, if input-conditioned target distributions p ( y | x ) "
    "are replaced with arbitrary distributions that have the same conditional expected values. "
    "This implies that we can, in principle, corrupt the training targets of a neural network "
    "with zero-mean noise without changing what the network learns. "
    "Combining this with the corrupted inputs from Equation 1, we obtain the empirical risk minimization task."
)
CLAIM = "在 L2 损失下，只要带噪目标与干净目标的条件期望相同，替换训练目标就不会改变最优参数。"


def _context(tmp_path: Path):
    path = tmp_path / "noise2noise.en.md"
    path.write_text("<!-- kb_page: 2 -->\n" + BODY, encoding="utf8")
    b = load_source_blocks(path)[0]
    row = {k: b.get(k) for k in ("block_id", "anchor_id", "heading_path", "page_start", "page_end")}
    row["text"] = BODY  # Actual coalesced passage lost its direct flag.
    hit = {"text": BODY, "meta": {**row, "source_path": str(path), "direct_question_evidence": True, "source_passages": [row]}}
    slot = {**row, "source_path": str(path), "preferred_system": "system_a", "candidate_hits": [1], "evidence_quote": BODY[:250] + "..."}
    return hit, {"slots": [slot]}


def test_existing_selected_direct_block_keeps_full_loss_context(tmp_path):
    hit, plan = _context(tmp_path)
    plan = attach_direct_source_evidence(plan, [hit])
    slot = plan["slots"][0]
    assert slot["evidence_quote"].startswith("input-conditioned target distributions")
    assert slot["source_evidence_quotes"] == [BODY]
    overlaid = _claim_evidence_hits_with_citation_plan([hit], plan)
    assert "L 2 minimization" in _hit_payload(overlaid[0])
    assert _support_score(CLAIM, _hit_payload(overlaid[0])) >= 5
    answer, _ = audit_and_repair_claim_evidence(CLAIM + " [1]", overlaid, allowed_citation_numbers={1}, drop_unsupported_high_risk_claims=True, enforce_user_visible_binding=True)
    assert "干净目标" in answer and "[1]" in answer


@pytest.mark.parametrize("key,bad", [("block_id", "wrong"), ("page_start", 8), ("anchor_id", "wrong")])
def test_nonmatching_primary_identity_cannot_promote_an_unmarked_passage(tmp_path, key, bad):
    hit, plan = _context(tmp_path)
    hit["meta"][key] = bad
    result = attach_direct_source_evidence(plan, [hit])
    assert not result["slots"][0].get("source_evidence_quotes")


@pytest.mark.parametrize("key,bad", [("page_start", 8), ("anchor_id", "wrong"), ("heading_path", "wrong")])
def test_existing_slot_cannot_lend_another_locator_the_complete_context(tmp_path, key, bad):
    hit, plan = _context(tmp_path)
    plan["slots"][0][key] = bad
    result = attach_direct_source_evidence(plan, [hit])
    assert not result["slots"][0].get("source_evidence_quotes")


def test_same_loss_condition_is_supported_in_english_and_chinese():
    english = "Under L2 minimization, replacing training targets with the same conditional expected values leaves the optimal parameters unchanged."
    assert _support_score(english, BODY) >= 5
    assert _support_score(english + " We combine this with corrupted inputs.", BODY) >= 5
    assert _support_score(CLAIM, BODY) >= 5


@pytest.mark.parametrize("claim", [
    "Under L2 minimization, input-conditioned target distributions can be changed to arbitrary distributions with the same conditional expected values.",
    "The input-conditioned target distributions change while the conditional expected values of the targets remain the same, under L2 minimization.",
])
def test_changing_target_distributions_is_not_changing_parameters(claim):
    assert _support_score(claim, BODY) >= 5


@pytest.mark.parametrize("change", [
    lambda c: c.replace("L2", "L1"),
    lambda c: c.replace("L2", "L3"),
    lambda c: c.replace("条件期望相同", "条件期望不同"),
    lambda c: c.replace("条件期望相同", "条件期望不相同"),
    lambda c: c.replace("条件期望", "无条件期望"),
    lambda c: c.replace("训练目标", "训练输入"),
    lambda c: c.replace("不会改变", "会改变"),
    lambda c: c + "噪声均值为2仍满足上述条件。",
    lambda c: "Under L2 minimization, we replace training targets with the same conditional expected values while changing the optimal parameters.",
    lambda c: "Under L2 minimization, the optimal network parameters change when training targets have the same conditional expected values.",
    lambda c: c.replace("替换训练目标就不会改变最优参数", "替换训练目标后最优参数发生改变"),
    lambda c: "Under L2 minimization, training targets have unchanged conditional expected values, but the optimal network parameters change.",
])
def test_full_context_does_not_lend_unrelated_numbers_or_reverse_the_relation(change):
    bad = change(CLAIM)
    assert not explicit_claim_relations_covered(bad, BODY)
    assert _support_score(bad, BODY) == 0


@pytest.mark.parametrize("evidence", [
    BODY.replace("L 2 minimization", "L 1 minimization"),
    BODY.replace("same conditional expected values", "different conditional expected values"),
    BODY.replace("the same conditional expected values", "not the same conditional expected values"),
    BODY.replace("input-conditioned target distributions", "target-conditioned input distributions"),
    "N2V excludes the target pixel from its receptive field and trains without clean targets.",
])
def test_changed_or_other_source_cannot_support_the_target_condition(evidence):
    assert _support_score(CLAIM, evidence) == 0
