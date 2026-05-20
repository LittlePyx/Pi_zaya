from __future__ import annotations

from pathlib import Path

import kb.generation_answer_finalize_runtime as finalize_runtime


def _identity_validate(answer: str, **_kwargs):
    return answer, {"ok": True}


def test_finalize_generation_answer_rewrites_template_only_citation_lookup(tmp_path) -> None:
    source_path = tmp_path / "paper.en.md"
    source_path.write_text("# Paper", encoding="utf-8")
    support = [
        {
            "source_path": str(source_path),
            "heading_path": "2. Related Work",
            "locate_anchor": "Most existing methods employ alternating direction method of multipliers (ADMM) [4].",
            "resolved_ref_num": 4,
        }
    ]

    result = finalize_runtime._finalize_generation_answer(
        "The paper cites [4] for this point.",
        prompt="ADMM 是怎么来的？作者是不是借鉴了以前的想法？",
        prompt_for_user="ADMM 是怎么来的？作者是不是借鉴了以前的想法？",
        answer_hits=[
            {
                "text": "Most existing methods employ alternating direction method of multipliers (ADMM) [4].",
                "meta": {"source_path": str(source_path), "heading_path": "2. Related Work"},
            }
        ],
        db_dir=Path("db"),
        locked_citation_source={"source_path": str(source_path)},
        answer_intent="reading",
        answer_depth="L2",
        answer_output_mode="fact_answer",
        paper_guide_mode=True,
        paper_guide_contract_enabled=False,
        paper_guide_prompt_family="citation_lookup",
        paper_guide_special_focus_block="",
        paper_guide_focus_source_path="",
        paper_guide_direct_source_path="",
        paper_guide_bound_source_path=str(source_path),
        paper_guide_candidate_refs_by_source={},
        paper_guide_support_slots=[],
        paper_guide_evidence_cards=[],
        paper_guide_contracts_seed={},
        paper_guide_retrieval_confidence_hint={},
        apply_paper_guide_answer_postprocess=lambda answer, **_kwargs: (answer, support),
        maybe_append_library_figure_markdown=lambda answer, **_kwargs: answer,
        validate_structured_citations=_identity_validate,
    )

    answer = str(result.get("answer") or "")
    assert "The paper cites" not in answer
    assert "ADMM" in answer
    assert "2. Related Work" in answer
    assert "[[CITE:" in answer
    assert ":4]]" in answer
    assert result["answer_quality"]["minimum_ok"] is True
    assert result["answer_quality"]["template_repair"]["changed"] is True


def test_finalize_generation_answer_surfaces_system_b_for_ordinary_question(tmp_path) -> None:
    source_path = tmp_path / "paper.en.md"
    source_path.write_text("# Paper", encoding="utf-8")
    source = str(source_path)
    support_slots = [
        {
            "source_path": source,
            "sid": "s1234abcd",
            "heading_path": "2. Related Work",
            "snippet": (
                "Most existing methods employ alternating direction method of multipliers "
                "(ADMM) [4]. ADMM-Net [21] unfolds this optimization idea into a network."
            ),
            "candidate_refs": [4, 21],
            "claim_type": "prior_work",
            "cite_policy": "prefer_ref",
        }
    ]

    def _validate_keep(answer: str, **kwargs):
        refs_by_source = kwargs.get("paper_guide_candidate_refs_by_source") or {}
        assert refs_by_source.get(source) == [4, 21]
        assert "[[CITE:s1234abcd:4]]" in answer
        assert "[[CITE:s1234abcd:21]]" not in answer
        return answer, {"raw_count": 1, "kept": 1, "rewritten": 0, "dropped": 0}

    result = finalize_runtime._finalize_generation_answer(
        "ADMM is not presented as this paper's original invention; it is used as prior optimization machinery.",
        prompt="I am new to this. Is ADMM original to this paper, or does it come from earlier work?",
        prompt_for_user="I am new to this. Is ADMM original to this paper, or does it come from earlier work?",
        answer_hits=[
            {
                "text": support_slots[0]["snippet"],
                "meta": {"source_path": source, "heading_path": "2. Related Work"},
            }
        ],
        db_dir=Path("db"),
        locked_citation_source={"source_path": source, "sid": "s1234abcd"},
        answer_intent="reading",
        answer_depth="L2",
        answer_output_mode="fact_answer",
        paper_guide_mode=True,
        paper_guide_contract_enabled=False,
        paper_guide_prompt_family="overview",
        paper_guide_special_focus_block="",
        paper_guide_focus_source_path="",
        paper_guide_direct_source_path="",
        paper_guide_bound_source_path=source,
        paper_guide_candidate_refs_by_source={},
        paper_guide_support_slots=support_slots,
        paper_guide_evidence_cards=[],
        paper_guide_contracts_seed={},
        paper_guide_retrieval_confidence_hint={},
        apply_paper_guide_answer_postprocess=lambda answer, **_kwargs: (answer, []),
        maybe_append_library_figure_markdown=lambda answer, **_kwargs: answer,
        validate_structured_citations=_validate_keep,
    )

    answer = str(result.get("answer") or "")
    assert "citation trail" not in answer
    assert "[[CITE:s1234abcd:4]]" in answer
    assert "[[CITE:s1234abcd:21]]" not in answer
    assert result["answer_quality"]["reference_opportunities"]["mode"] == "inline"
    assert result["answer_quality"]["reference_opportunities"]["injected_refs"] == [4]
    assert result["answer_quality"]["reference_opportunities"]["refs"][:2] == [4, 21]
