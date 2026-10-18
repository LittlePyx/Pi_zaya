from __future__ import annotations

import pytest

from kb.claim_evidence_runtime import audit_and_repair_claim_evidence


def _strict(answer: str) -> tuple[str, dict]:
    return audit_and_repair_claim_evidence(
        answer, [], allow_citation_repairs=False,
        allowed_citation_numbers=set(), drop_unsupported_high_risk_claims=True,
    )


@pytest.mark.parametrize("answer", [
    "没有这项随机临床试验。",
    "**没有这项随机临床试验。**",
    "不存在该项试验。",
    "该试验不存在。",
    "就目前证据而言，该试验不存在。",
])
def test_short_absolute_absence_does_not_escape_evidence_gate(answer: str) -> None:
    repaired, meta = _strict(answer)
    assert not repaired.strip(" *")
    assert meta["dropped_unsupported_unplanned_claims"] == 1


@pytest.mark.parametrize("answer", [
    "这只能说明所引片段不支持该问题，不能据此断言两篇论文全文完全没有临床内容。",
    "不能据此断言两篇论文全文完全没有临床内容。",
    "仅凭当前检索片段，无法确认所问随机临床试验，也不能提供患者人数。",
    "当前引用证据不足以证明这两种方法的普遍排名。",
    "依据本轮检索证据，不能给出诊断敏感度差值。",
])
def test_limits_on_full_text_inference_are_not_paper_absence_claims(answer: str) -> None:
    repaired, meta = _strict(answer)
    assert repaired == answer
    assert meta["dropped_unsupported_unplanned_claims"] == 0


@pytest.mark.parametrize("extra", [
    "但实验的诊断敏感度提高了80%。",
    "因此证明该方法优于所有模型。",
    "该试验报告120名患者。",
])
def test_a_retrieval_caveat_cannot_license_added_facts(extra: str) -> None:
    boundary = "不能据此断言两篇论文全文完全没有临床内容。"
    repaired, meta = _strict(boundary + extra)
    assert boundary in repaired
    assert extra not in repaired
    assert meta["dropped_unsupported_unplanned_claims"] >= 1


@pytest.mark.parametrize("answer", [
    "不能据此断言两篇论文全文完全没有临床内容，但实验的敏感度提高了80%。",
    "不能据此断言两篇论文全文没有报告120名患者。",
])
def test_boundary_wording_does_not_bypass_numeric_grounding(answer: str) -> None:
    repaired, meta = _strict(answer)
    assert not repaired
    assert meta["dropped_unsupported_unplanned_claims"] >= 1


def test_scoped_modal_prefix_does_not_license_an_affirmative_tail() -> None:
    answer = "当前引用证据不足以证明该结论，但该实验报告120名患者。"
    repaired, meta = _strict(answer)
    assert not repaired
    assert meta["dropped_unsupported_unplanned_claims"] == 1


@pytest.mark.parametrize(("answer", "retained"), [
    ("这两篇去噪论文只能作为方法背景，不能提供该试验数据。", ""),
    ("建议另外查找研究；这篇论文无法提供该试验数据。", "建议另外查找研究。"),
    ("阅读建议：该论文不能给出患者人数。", ""),
    ("当前选中的两篇去噪论文本身不提供该证据。", ""),
])
def test_reading_advice_cannot_assert_unverified_paper_inability(answer: str, retained: str) -> None:
    repaired, meta = _strict(answer)
    assert repaired == retained
    assert meta["dropped_unsupported_unplanned_claims"] == 1


@pytest.mark.parametrize("answer", [
    "结论：仅依据这两篇论文的检索内容，无法给出所问临床试验数据。",
    "仅凭所选文献的检索片段，不能提供该试验患者人数。",
    "当前检索内容无法给出所问试验结果。",
    "当前检索片段不提供所问试验结果。",
])
def test_retrieval_content_inability_keeps_its_explicit_scope(answer: str) -> None:
    repaired, meta = _strict(answer)
    assert repaired == answer
    assert meta["dropped_unsupported_unplanned_claims"] == 0
