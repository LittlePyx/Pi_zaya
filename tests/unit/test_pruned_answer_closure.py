from __future__ import annotations

import pytest

from kb.answer_contract import _build_answer_quality_probe
from kb.claim_evidence_runtime import audit_and_repair_claim_evidence
from kb.task_runtime import _finalize_generation_answer


EVIDENCE = (
    "In stark contrast to our approach, they need a perfect, explicit computational model of the corruption; "
    "our requirement of the knowledge of the appropriate statistical summary (mean, median) is significantly less limiting."
)
PROMPT = (
    "Does this method require a likelihood or an explicit model of the corruption process? "
    "What statistical condition remains necessary?"
)
SOURCE = "C:/papers/noise2noise/noise2noise.en.md"


def _quality(answer: str) -> dict:
    return _build_answer_quality_probe(
        answer, has_hits=True, contract_enabled=False, paper_guide_mode=True, prompt_family="method"
    )


@pytest.mark.parametrize("separator", [";", "；"])
@pytest.mark.parametrize("addition", [
    "it does not give a single named formal condition beyond the conditional-distribution assumption above.",
    "DIP achieves 99 dB on all targets [1].",
    "Noise2Self requires correlated noise with variance 55 [1].",
])
def test_pruned_final_clause_does_not_leave_a_false_incomplete_answer(separator, addition):
    raw = f"The appropriate statistical summary (mean, median) must be known [1]{separator} {addition}"
    repaired, metadata = audit_and_repair_claim_evidence(
        raw, [{"text": EVIDENCE, "meta": {"source_path": SOURCE}}], prompt=PROMPT,
        allowed_citation_numbers={1}, drop_unsupported_high_risk_claims=True,
        enforce_user_visible_binding=True,
    )
    assert "mean, median" in repaired and "[1]" in repaired
    assert addition not in repaired
    assert metadata["dropped_unsupported_unplanned_claims"] + metadata["dropped_hard_mismatch_claims"] > 0
    assert _quality(repaired)["minimum_ok"] is True
    assert not repaired.rstrip().endswith((";", "；"))


def test_authored_trailing_semicolon_is_still_incomplete():
    raw = "The appropriate statistical summary (mean, median) must be known [1];"
    repaired, _ = audit_and_repair_claim_evidence(
        raw, [{"text": EVIDENCE}], prompt=PROMPT, drop_unsupported_high_risk_claims=True,
    )
    assert repaired.endswith(";")
    assert _quality(repaired)["minimum_ok"] is False


def test_pruning_preserves_math_semicolons_and_surviving_clauses():
    raw = "The network uses $Y = [Y^c;Y^p]$ [1]; The method uses two branches [1]."
    repaired, _ = audit_and_repair_claim_evidence(
        raw, [{"text": "The network uses $Y = [Y^c;Y^p]$. The method uses two branches."}],
        drop_unsupported_high_risk_claims=True,
    )
    assert "$Y = [Y^c;Y^p]$" in repaired
    assert "[1]; The method uses two branches" in repaired


@pytest.mark.parametrize("source,evidence", [
    (SOURCE, EVIDENCE),
    ("C:/papers/noise2self/noise2self.en.md", "Noise2Self calibrates median filters and non-local means."),
])
def test_n2n_r17_claim_shape_through_real_finalizer(source, evidence):
    # Reconstruct the recorded claim shape; the original r17 raw draft was not saved.
    raw = (
        "No — the method does not require an explicit model of the corruption process [1]. "
        "The paper contrasts its approach with methods that need a perfect, explicit computational model of the corruption; "
        "our requirement is knowledge of the appropriate statistical summary (mean, median), "
        "rather than requiring a full likelihood or corruption model [1].\n\n"
        'The same property is noted in the experiments: DIP "shares the property that neither clean examples '
        'nor an explicit model of the corruption is needed." [1]\n\n'
        "The appropriate statistical summary must be known for the corrupted target distribution [1]; "
        "it does not give a single named formal condition beyond the conditional-distribution assumption above."
    )
    plan = {"slots": [{
        "preferred_system": "system_a", "candidate_hits": [1], "source_path": source,
        "source_name": "noise2noise" if source == SOURCE else "noise2self", "evidence_quote": evidence,
    }]}
    out = _finalize_generation_answer(
        raw, prompt=PROMPT, prompt_for_user=PROMPT,
        answer_hits=[{"text": evidence, "meta": {
            "source_path": source,
            "heading_path": "Noise2Noise / 4. Discussion" if source == SOURCE else "Noise2Self / Abstract",
        }}],
        db_dir=None, locked_citation_source=None,
        answer_intent="reading", answer_depth="L2", answer_output_mode="reading_guide",
        paper_guide_mode=True, paper_guide_contract_enabled=False, paper_guide_prompt_family="method",
        paper_guide_special_focus_block="", paper_guide_focus_source_path=source,
        paper_guide_direct_source_path=source, paper_guide_bound_source_path=source,
        paper_guide_candidate_refs_by_source={}, paper_guide_support_slots=[], paper_guide_evidence_cards=[],
        paper_guide_contracts_seed={"citation_plan": plan},
    )
    answer = out["answer"]
    assert "DIP" not in answer
    assert "does not give a single named formal condition" not in answer
    if source == SOURCE:
        assert out["answer_quality"]["minimum_ok"] is True
        assert out["answer_quality"]["claim_evidence"]["minimum_ok"] is True
        assert "target" in answer and "mean" in answer and "median" in answer and "[1]" in answer
    else:
        assert "does not require an explicit model" not in answer
