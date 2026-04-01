import kb.generation_answer_finalize_runtime as finalize_runtime


def test_finalize_generation_answer_runs_postprocess_validate_and_quality(monkeypatch):
    calls = []
    figure_kwargs = {}

    monkeypatch.setattr(finalize_runtime, "_reconcile_kb_notice", lambda answer, **kwargs: calls.append("reconcile") or (answer + " [reconcile]"))
    monkeypatch.setattr(finalize_runtime, "_apply_answer_contract_v1", lambda answer, **kwargs: calls.append("contract") or (answer + " [contract]"))
    monkeypatch.setattr(finalize_runtime, "_enhance_kb_miss_fallback", lambda answer, **kwargs: calls.append("enhance") or (answer + " [enhance]"))
    monkeypatch.setattr(finalize_runtime, "_build_answer_quality_probe", lambda answer, **kwargs: {"minimum_ok": True, "answer": answer})

    def _figure(answer, **kwargs):
        figure_kwargs.update(kwargs)
        calls.append("figure")
        return answer + " [figure]"

    out = finalize_runtime._finalize_generation_answer(
        "raw [[CITE:s1]]",
        prompt="How is APR grounded?",
        prompt_for_user="How is APR grounded?",
        answer_hits=[{"meta": {"source_path": "demo.md"}}],
        db_dir="db",
        locked_citation_source={"sid": "s123", "source_name": "demo.pdf"},
        answer_intent="reading",
        answer_depth="medium",
        answer_output_mode="reading_guide",
        paper_guide_mode=True,
        paper_guide_contract_enabled=True,
        paper_guide_prompt_family="method",
        paper_guide_special_focus_block="focus",
        paper_guide_focus_source_path="focus.md",
        paper_guide_direct_source_path="direct.md",
        paper_guide_bound_source_path="bound.md",
        paper_guide_candidate_refs_by_source={"demo.md": [35]},
        paper_guide_support_slots=[{"support_example": "[[SUPPORT:DOC-1]]"}],
        paper_guide_evidence_cards=[{"doc_idx": 1}],
        apply_paper_guide_answer_postprocess=lambda answer, **kwargs: (calls.append("postprocess") or (answer + " [post]", [{"line_index": 0}])),
        maybe_append_library_figure_markdown=_figure,
        validate_structured_citations=lambda answer, **kwargs: (calls.append("validate") or (answer + " [validated]", {"kept": 1})),
    )

    assert calls == ["reconcile", "contract", "enhance", "postprocess", "figure", "validate"]
    assert figure_kwargs["bound_source_path"] == "bound.md"
    assert out["answer"].endswith("[validated]")
    assert out["paper_guide_support_resolution"] == [{"line_index": 0}]
    assert out["citation_validation"] == {"kept": 1}
    assert out["answer_quality"]["minimum_ok"] is True


def test_finalize_generation_answer_skips_contract_when_disabled(monkeypatch):
    calls = []

    monkeypatch.setattr(finalize_runtime, "_reconcile_kb_notice", lambda answer, **kwargs: answer)
    monkeypatch.setattr(finalize_runtime, "_apply_answer_contract_v1", lambda answer, **kwargs: calls.append("contract") or answer)
    monkeypatch.setattr(finalize_runtime, "_enhance_kb_miss_fallback", lambda answer, **kwargs: answer)
    monkeypatch.setattr(finalize_runtime, "_build_answer_quality_probe", lambda answer, **kwargs: {"minimum_ok": True})

    out = finalize_runtime._finalize_generation_answer(
        "raw",
        prompt="Explain Figure 1.",
        prompt_for_user="Explain Figure 1.",
        answer_hits=[],
        db_dir=None,
        locked_citation_source=None,
        answer_intent="reading",
        answer_depth="medium",
        answer_output_mode="reading_guide",
        paper_guide_mode=False,
        paper_guide_contract_enabled=False,
        paper_guide_prompt_family="figure_walkthrough",
        paper_guide_special_focus_block="",
        paper_guide_focus_source_path="",
        paper_guide_direct_source_path="",
        paper_guide_bound_source_path="",
        paper_guide_candidate_refs_by_source={},
        paper_guide_support_slots=[],
        paper_guide_evidence_cards=[],
        apply_paper_guide_answer_postprocess=lambda answer, **kwargs: (answer, []),
        maybe_append_library_figure_markdown=lambda answer, **kwargs: answer,
        validate_structured_citations=lambda answer, **kwargs: (answer, {}),
    )

    assert calls == []
    assert out["answer"] == "raw"


def test_finalize_generation_answer_resanitizes_overview_after_citation_validation_when_family_is_inferred(monkeypatch):
    monkeypatch.setattr(finalize_runtime, "_reconcile_kb_notice", lambda answer, **kwargs: answer)
    monkeypatch.setattr(finalize_runtime, "_apply_answer_contract_v1", lambda answer, **kwargs: answer)
    monkeypatch.setattr(finalize_runtime, "_enhance_kb_miss_fallback", lambda answer, **kwargs: answer)
    monkeypatch.setattr(finalize_runtime, "_build_answer_quality_probe", lambda answer, **kwargs: {"minimum_ok": True, "answer": answer})

    out = finalize_runtime._finalize_generation_answer(
        "The authors report improved throughput.",
        prompt="What throughput contribution do the authors claim?",
        prompt_for_user="What throughput contribution do the authors claim?",
        answer_hits=[{"meta": {"source_path": "demo.md"}}],
        db_dir="db",
        locked_citation_source={"sid": "s123", "source_name": "demo.pdf"},
        answer_intent="reading",
        answer_depth="medium",
        answer_output_mode="reading_guide",
        paper_guide_mode=True,
        paper_guide_contract_enabled=False,
        paper_guide_prompt_family="",
        paper_guide_special_focus_block="",
        paper_guide_focus_source_path="focus.md",
        paper_guide_direct_source_path="direct.md",
        paper_guide_bound_source_path="bound.md",
        paper_guide_candidate_refs_by_source={"demo.md": [26]},
        paper_guide_support_slots=[],
        paper_guide_evidence_cards=[],
        apply_paper_guide_answer_postprocess=lambda answer, **kwargs: (answer, []),
        maybe_append_library_figure_markdown=lambda answer, **kwargs: answer,
        validate_structured_citations=lambda answer, **kwargs: (
            answer + " [[CITE:s1234abcd:26]]",
            {"kept": 1},
        ),
    )

    assert "[[CITE:" not in out["answer"]
    assert "throughput" in out["answer"].lower()
    assert out["answer_quality"]["answer"] == out["answer"]
