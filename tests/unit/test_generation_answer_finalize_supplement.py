import kb.generation_answer_finalize_runtime as finalize_runtime


def test_finalize_generation_answer_appends_dual_layer_supplement_block_on_low_confidence():
    out = finalize_runtime._finalize_generation_answer(
        "Core claim from retrieved evidence.",
        prompt="How should I understand this method?",
        prompt_for_user="How should I understand this method?",
        answer_hits=[{"meta": {"source_path": "demo.md"}}],
        db_dir="db",
        locked_citation_source=None,
        answer_intent="reading",
        answer_depth="medium",
        answer_output_mode="reading_guide",
        paper_guide_mode=True,
        paper_guide_contract_enabled=False,
        paper_guide_prompt_family="method",
        paper_guide_special_focus_block="",
        paper_guide_focus_source_path="focus.md",
        paper_guide_direct_source_path="direct.md",
        paper_guide_bound_source_path="bound.md",
        paper_guide_candidate_refs_by_source={},
        paper_guide_support_slots=[],
        paper_guide_evidence_cards=[],
        paper_guide_retrieval_confidence_hint={
            "low_confidence": True,
            "low_confidence_reason": "strict_family_sparse_hits",
        },
        apply_paper_guide_answer_postprocess=lambda answer, **kwargs: (answer, []),
        maybe_append_library_figure_markdown=lambda answer, **kwargs: answer,
        validate_structured_citations=lambda answer, **kwargs: (answer, {}),
    )

    text = str(out.get("answer") or "")
    assert "Supplementary note (generic knowledge, non-retrieved content" in text
    assert "not explicit claims from the paper" in text
    assert "Core claim from retrieved evidence" in text


def test_finalize_generation_answer_does_not_append_supplement_when_user_opted_out():
    out = finalize_runtime._finalize_generation_answer(
        "Core claim from retrieved evidence.",
        prompt="Only from the paper, no supplement.",
        prompt_for_user="Only from the paper, no supplement.",
        answer_hits=[{"meta": {"source_path": "demo.md"}}],
        db_dir="db",
        locked_citation_source=None,
        answer_intent="reading",
        answer_depth="medium",
        answer_output_mode="reading_guide",
        paper_guide_mode=True,
        paper_guide_contract_enabled=False,
        paper_guide_prompt_family="method",
        paper_guide_special_focus_block="",
        paper_guide_focus_source_path="focus.md",
        paper_guide_direct_source_path="direct.md",
        paper_guide_bound_source_path="bound.md",
        paper_guide_candidate_refs_by_source={},
        paper_guide_support_slots=[],
        paper_guide_evidence_cards=[],
        paper_guide_retrieval_confidence_hint={
            "low_confidence": True,
            "low_confidence_reason": "strict_family_sparse_hits",
        },
        apply_paper_guide_answer_postprocess=lambda answer, **kwargs: (answer, []),
        maybe_append_library_figure_markdown=lambda answer, **kwargs: answer,
        validate_structured_citations=lambda answer, **kwargs: (answer, {}),
    )

    text = str(out.get("answer") or "")
    assert "Supplementary note (generic knowledge, non-retrieved content" not in text
    assert "补充说明（通用知识，非检索片段内容" not in text


def test_finalize_generation_answer_prefers_dynamic_supplement_lines_when_available():
    calls = []

    out = finalize_runtime._finalize_generation_answer(
        "Core claim from retrieved evidence.",
        prompt="How should I understand this method?",
        prompt_for_user="How should I understand this method?",
        answer_hits=[{"meta": {"source_path": "demo.md"}}],
        db_dir="db",
        locked_citation_source=None,
        answer_intent="reading",
        answer_depth="medium",
        answer_output_mode="reading_guide",
        paper_guide_mode=True,
        paper_guide_contract_enabled=False,
        paper_guide_prompt_family="method",
        paper_guide_special_focus_block="",
        paper_guide_focus_source_path="focus.md",
        paper_guide_direct_source_path="direct.md",
        paper_guide_bound_source_path="bound.md",
        paper_guide_candidate_refs_by_source={},
        paper_guide_support_slots=[],
        paper_guide_evidence_cards=[],
        paper_guide_retrieval_confidence_hint={
            "low_confidence": True,
            "low_confidence_reason": "strict_family_sparse_hits",
        },
        apply_paper_guide_answer_postprocess=lambda answer, **kwargs: (answer, [{"heading_path": "Method"}]),
        maybe_append_library_figure_markdown=lambda answer, **kwargs: answer,
        validate_structured_citations=lambda answer, **kwargs: (answer, {}),
        build_paper_guide_supplement_lines=lambda **kwargs: calls.append(kwargs) or (
            "- From a general method perspective, this kind of module usually trades extra constraints for stability.\n"
            "- That interpretation is AI-added background rather than a direct paper claim."
        ),
    )

    text = str(out.get("answer") or "")
    assert len(calls) == 1
    assert "trades extra constraints for stability" in text
    assert "not explicit claims from the paper" in text
    assert "Method understanding is more reliable when input/output" not in text


def test_finalize_generation_answer_dynamic_supplement_falls_back_to_default_when_builder_returns_empty():
    out = finalize_runtime._finalize_generation_answer(
        "Core claim from retrieved evidence.",
        prompt="How should I understand this method?",
        prompt_for_user="How should I understand this method?",
        answer_hits=[{"meta": {"source_path": "demo.md"}}],
        db_dir="db",
        locked_citation_source=None,
        answer_intent="reading",
        answer_depth="medium",
        answer_output_mode="reading_guide",
        paper_guide_mode=True,
        paper_guide_contract_enabled=False,
        paper_guide_prompt_family="method",
        paper_guide_special_focus_block="",
        paper_guide_focus_source_path="focus.md",
        paper_guide_direct_source_path="direct.md",
        paper_guide_bound_source_path="bound.md",
        paper_guide_candidate_refs_by_source={},
        paper_guide_support_slots=[],
        paper_guide_evidence_cards=[],
        paper_guide_retrieval_confidence_hint={
            "low_confidence": True,
            "low_confidence_reason": "strict_family_sparse_hits",
        },
        apply_paper_guide_answer_postprocess=lambda answer, **kwargs: (answer, []),
        maybe_append_library_figure_markdown=lambda answer, **kwargs: answer,
        validate_structured_citations=lambda answer, **kwargs: (answer, {}),
        build_paper_guide_supplement_lines=lambda **kwargs: "",
    )

    text = str(out.get("answer") or "")
    assert "Method understanding is more reliable when input/output" in text


def test_finalize_generation_answer_appends_supplement_for_sparse_negative_shell_without_low_conf_hint():
    out = finalize_runtime._finalize_generation_answer(
        "The retrieved paper does not specify the implementation detail.",
        prompt="How is this module implemented?",
        prompt_for_user="How is this module implemented?",
        answer_hits=[{"meta": {"source_path": "demo.md"}}],
        db_dir="db",
        locked_citation_source=None,
        answer_intent="reading",
        answer_depth="medium",
        answer_output_mode="reading_guide",
        paper_guide_mode=True,
        paper_guide_contract_enabled=False,
        paper_guide_prompt_family="method",
        paper_guide_special_focus_block="",
        paper_guide_focus_source_path="focus.md",
        paper_guide_direct_source_path="direct.md",
        paper_guide_bound_source_path="bound.md",
        paper_guide_candidate_refs_by_source={},
        paper_guide_support_slots=[],
        paper_guide_evidence_cards=[],
        apply_paper_guide_answer_postprocess=lambda answer, **kwargs: (answer, []),
        maybe_append_library_figure_markdown=lambda answer, **kwargs: answer,
        validate_structured_citations=lambda answer, **kwargs: (answer, {}),
        build_paper_guide_supplement_lines=lambda **kwargs: "- In many papers, this kind of implementation detail is omitted when it is not the paper's main novelty.",
    )

    text = str(out.get("answer") or "")
    assert "Supplementary note (generic knowledge, non-retrieved content" in text
    assert "this kind of implementation detail is omitted" in text
