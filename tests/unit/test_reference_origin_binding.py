from __future__ import annotations

import pytest

from kb.generation_answer_finalize_runtime import _finalize_fast_exact_generation_answer
from kb.generation_answer_finalize_runtime import _relocate_planned_origin_system_b_markers


def test_origin_reference_moves_to_explicit_negative_claim_not_generic_method_use():
    marker = "[[CITE:source:4]]"
    answer = f"ADMM不是这篇论文作者发明的。\n\n原文提到已有方法采用 ADMM {marker}。\n\nADMM 的角色是既有优化方法。"
    plan = {"slots": [{"preferred_system": "system_b", "claim_type": "origin", "topic": "ADMM", "sid": "source", "candidate_refs": [4]}]}
    result = _relocate_planned_origin_system_b_markers(answer, citation_plan=plan)
    assert marker in result.splitlines()[0]
    assert result.count(marker) == 1


def test_other_method_negative_claim_cannot_take_planned_origin_marker():
    marker = "[[CITE:source:4]]"
    answer = f"OtherMethod不是本文作者发明的。\n\nADMM comes from prior work {marker}."
    plan = {"slots": [{"preferred_system": "system_b", "claim_type": "origin", "topic": "ADMM", "sid": "source", "candidate_refs": [4]}]}
    assert _relocate_planned_origin_system_b_markers(answer, citation_plan=plan) == answer


def test_origin_passage_topic_resolves_its_unique_method_identifier():
    marker = "[[CITE:source:4]]"
    answer = f"不是作者发明的。\n\n已有方法采用 ADMM {marker}。"
    plan = {"slots": [{"preferred_system": "system_b", "claim_type": "origin", "topic": "most of the existing methods employ alternating direction method of multipliers (ADMM) [4],", "sid": "source", "candidate_refs": [4]}]}
    result = _relocate_planned_origin_system_b_markers(answer, citation_plan=plan)
    assert result.splitlines()[0] == f"ADMM不是作者发明的 {marker}。"


def test_fast_answer_binds_origin_claim_before_reference_identity_validation():
    marker = "[[CITE:source:4]]"
    plan = {"budget": {"system_a": 1, "system_b": 1}, "slots": [{"preferred_system": "system_b", "claim_type": "origin", "topic": "ADMM", "sid": "source", "candidate_refs": [4]}]}
    received = []

    def validate(answer, **_kwargs):
        received.append(answer)
        raise RuntimeError("validation boundary reached")

    with pytest.raises(RuntimeError, match="validation boundary reached"):
        _finalize_fast_exact_generation_answer(
            f"ADMM不是本文作者发明的。\n\n已有方法采用 ADMM {marker}。",
            prompt="ADMM 是本文发明的吗？", prompt_for_user="ADMM 是本文发明的吗？",
            answer_hits=[], db_dir=None, locked_citation_source=None,
            answer_intent="reading", answer_depth="L2", answer_output_mode="reading_guide",
            paper_guide_prompt_family="citation_lookup", paper_guide_bound_source_path="",
            paper_guide_candidate_refs_by_source={}, paper_guide_support_slots=[],
            paper_guide_evidence_cards=[], paper_guide_precomputed_support_resolution=[],
            paper_guide_contracts_seed={"citation_plan": plan},
            paper_guide_retrieval_confidence_hint={}, research_answer_plan="",
            validate_structured_citations=validate,
        )
    assert marker in received[0].splitlines()[0]
    assert received[0].count(marker) == 1
