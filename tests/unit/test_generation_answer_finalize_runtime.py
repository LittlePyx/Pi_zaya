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
    assert "[validated]" in out["answer"]
    assert out["paper_guide_support_resolution"] == [{"line_index": 0}]
    assert out["citation_validation"] == {"kept": 1}
    assert out["answer_quality"]["minimum_ok"] is True


def test_finalize_generation_answer_passes_shared_primary_evidence_into_answer_contract(monkeypatch):
    seen = {}

    monkeypatch.setattr(finalize_runtime, "_reconcile_kb_notice", lambda answer, **kwargs: answer)
    monkeypatch.setattr(
        finalize_runtime,
        "_apply_answer_contract_v1",
        lambda answer, **kwargs: seen.update({"primary_evidence": dict(kwargs.get("primary_evidence") or {})}) or answer,
    )
    monkeypatch.setattr(finalize_runtime, "_enhance_kb_miss_fallback", lambda answer, **kwargs: answer)
    monkeypatch.setattr(finalize_runtime, "_build_answer_quality_probe", lambda answer, **kwargs: {"minimum_ok": True})

    out = finalize_runtime._finalize_generation_answer(
        "Core grounded answer.",
        prompt="How is Fourier single-pixel imaging discussed?",
        prompt_for_user="How is Fourier single-pixel imaging discussed?",
        answer_hits=[{"meta": {"source_path": "demo.md"}}],
        db_dir="db",
        locked_citation_source=None,
        answer_intent="reading",
        answer_depth="medium",
        answer_output_mode="reading_guide",
        paper_guide_mode=True,
        paper_guide_contract_enabled=True,
        paper_guide_prompt_family="overview",
        paper_guide_special_focus_block="",
        paper_guide_focus_source_path="focus.md",
        paper_guide_direct_source_path="direct.md",
        paper_guide_bound_source_path="bound.md",
        paper_guide_candidate_refs_by_source={},
        paper_guide_support_slots=[],
        paper_guide_evidence_cards=[
            {
                "doc_idx": 1,
                "primary_evidence": {
                    "source_name": "fallback.pdf",
                    "heading_path": "2.4 Efficiency",
                    "snippet": "Section 2.4 discusses efficiency only.",
                },
            }
        ],
        paper_guide_contracts_seed={
            "primary_evidence": {
                "source_name": "OE-2017.pdf",
                "heading_path": "2.2 Basis patterns generation",
                "snippet": "Section 2.2 discusses Fourier single-pixel imaging and compares it with Hadamard sampling.",
            }
        },
        apply_paper_guide_answer_postprocess=lambda answer, **kwargs: (answer, []),
        maybe_append_library_figure_markdown=lambda answer, **kwargs: answer,
        validate_structured_citations=lambda answer, **kwargs: (answer, {}),
    )

    assert out["answer"] == "Core grounded answer."
    assert seen["primary_evidence"]["source_name"] == "OE-2017.pdf"
    assert seen["primary_evidence"]["heading_path"] == "2.2 Basis patterns generation"


def test_finalize_generation_answer_passes_shared_primary_evidence_from_cards_for_non_paper_guide(monkeypatch):
    seen = {}

    monkeypatch.setattr(finalize_runtime, "_reconcile_kb_notice", lambda answer, **kwargs: answer)
    monkeypatch.setattr(
        finalize_runtime,
        "_apply_answer_contract_v1",
        lambda answer, **kwargs: seen.update({"primary_evidence": dict(kwargs.get("primary_evidence") or {})}) or answer,
    )
    monkeypatch.setattr(finalize_runtime, "_enhance_kb_miss_fallback", lambda answer, **kwargs: answer)
    monkeypatch.setattr(finalize_runtime, "_build_answer_quality_probe", lambda answer, **kwargs: {"minimum_ok": True})

    out = finalize_runtime._finalize_generation_answer(
        "Core grounded answer.",
        prompt="Which paper in my library directly compares Hadamard single-pixel imaging and Fourier single-pixel imaging?",
        prompt_for_user="Which paper in my library directly compares Hadamard single-pixel imaging and Fourier single-pixel imaging?",
        answer_hits=[{"meta": {"source_path": "demo.md"}}],
        db_dir="db",
        locked_citation_source=None,
        answer_intent="compare",
        answer_depth="medium",
        answer_output_mode="reading_guide",
        paper_guide_mode=False,
        paper_guide_contract_enabled=True,
        paper_guide_prompt_family="",
        paper_guide_special_focus_block="",
        paper_guide_focus_source_path="",
        paper_guide_direct_source_path="",
        paper_guide_bound_source_path="",
        paper_guide_candidate_refs_by_source={},
        paper_guide_support_slots=[],
        paper_guide_evidence_cards=[
            {
                "doc_idx": 1,
                "primary_evidence": {
                    "source_name": "OE-2017.pdf",
                    "heading_path": "2.2 Basis patterns generation",
                    "snippet": "Section 2.2 explicitly compares the two methods in terms of basis pattern properties.",
                    "block_id": "blk_22",
                },
            }
        ],
        apply_paper_guide_answer_postprocess=lambda answer, **kwargs: (answer, []),
        maybe_append_library_figure_markdown=lambda answer, **kwargs: answer,
        validate_structured_citations=lambda answer, **kwargs: (answer, {}),
    )

    assert out["answer"] == "Core grounded answer."
    assert seen["primary_evidence"]["source_name"] == "OE-2017.pdf"
    assert seen["primary_evidence"]["heading_path"] == "2.2 Basis patterns generation"
    assert seen["primary_evidence"]["block_id"] == "blk_22"
    contracts = out["paper_guide_contracts"]
    assert contracts["version"] == 1
    assert contracts["primary_evidence"]["heading_path"] == "2.2 Basis patterns generation"
    assert contracts["render_packet"]["primary_evidence"]["block_id"] == "blk_22"
    assert contracts["render_packet"]["answer_markdown"] == "Core grounded answer."


def test_finalize_generation_answer_builds_paper_guide_contract_snapshot(monkeypatch):
    monkeypatch.setattr(finalize_runtime, "_reconcile_kb_notice", lambda answer, **kwargs: answer)
    monkeypatch.setattr(finalize_runtime, "_apply_answer_contract_v1", lambda answer, **kwargs: answer)
    monkeypatch.setattr(finalize_runtime, "_enhance_kb_miss_fallback", lambda answer, **kwargs: answer)
    monkeypatch.setattr(finalize_runtime, "_build_answer_quality_probe", lambda answer, **kwargs: {"minimum_ok": True})

    out = finalize_runtime._finalize_generation_answer(
        "Core grounded answer.",
        prompt="How is APR grounded?",
        prompt_for_user="How is APR grounded?",
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
        paper_guide_candidate_refs_by_source={"demo.md": [35]},
        paper_guide_support_slots=[{"support_id": "slot-1", "source_path": "demo.md", "candidate_refs": ["35"]}],
        paper_guide_evidence_cards=[],
        paper_guide_contracts_seed={
            "prompt_context": {
                "target_scope": {"prompt_family": "method"},
                "focus_source_path": "focus.md",
                "bound_source_path": "bound.md",
            },
            "primary_evidence": {
                "source_name": "demo.pdf",
                "heading_path": "Methods / APR",
                "snippet": "APR uses phase correlation for registration.",
            },
        },
        apply_paper_guide_answer_postprocess=lambda answer, **kwargs: (
            answer,
            [
                {
                    "support_id": "DOC-1-S1",
                    "source_path": "demo.md",
                    "block_id": "b-7",
                    "anchor_id": "a-7",
                    "heading_path": "Methods / APR",
                    "claim_type": "method_claim",
                    "resolved_ref_num": "35",
                    "segment_text": "APR uses phase correlation for registration.",
                    "line_index": 0,
                }
            ],
        ),
        maybe_append_library_figure_markdown=lambda answer, **kwargs: answer,
        validate_structured_citations=lambda answer, **kwargs: (answer, {"kept": 1}),
    )

    contracts = out["paper_guide_contracts"]
    assert contracts["version"] == 1
    assert contracts["intent"]["family"] == "method"
    assert contracts["retrieval_bundle"]["prompt_family"] == "method"
    assert contracts["retrieval_bundle"]["candidate_refs_by_source"] == {"demo.md": [35]}
    assert contracts["support_pack"]["family"] == "method"
    assert contracts["support_pack"]["answer_markdown"] == "Core grounded answer."
    assert contracts["support_pack"]["support_records"][0]["resolved_ref_num"] == 35
    assert contracts["grounding_trace"][0]["segment_id"] == "DOC-1-S1"
    assert contracts["grounding_trace"][0]["text"] == "APR uses phase correlation for registration."
    assert contracts["grounding_trace"][0]["primary_block_id"] == "b-7"
    assert contracts["render_packet"]["answer_markdown"] == "Core grounded answer."
    assert contracts["render_packet"]["citation_validation"] == {"kept": 1}
    assert contracts["render_packet"]["primary_evidence"]["heading_path"] == "Methods / APR"
    assert contracts["prompt_context"]["target_scope"]["prompt_family"] == "method"
    assert contracts["prompt_context"]["focus_source_path"] == "focus.md"
    assert contracts["primary_evidence"]["heading_path"] == "Methods / APR"


def test_finalize_generation_answer_contract_snapshot_falls_back_to_support_slots():
    out = finalize_runtime._finalize_generation_answer(
        "Core grounded answer.",
        prompt="Explain Figure 3 panel F.",
        prompt_for_user="Explain Figure 3 panel F.",
        answer_hits=[{"meta": {"source_path": "demo.md"}}],
        db_dir="db",
        locked_citation_source=None,
        answer_intent="reading",
        answer_depth="medium",
        answer_output_mode="reading_guide",
        paper_guide_mode=True,
        paper_guide_contract_enabled=False,
        paper_guide_prompt_family="figure_walkthrough",
        paper_guide_special_focus_block="",
        paper_guide_focus_source_path="focus.md",
        paper_guide_direct_source_path="direct.md",
        paper_guide_bound_source_path="bound.md",
        paper_guide_candidate_refs_by_source={},
        paper_guide_support_slots=[
            {
                "support_id": "slot-figure-1",
                "source_path": "demo.md",
                "figure_number": "3",
                "panel_letters": ["F", "f"],
            }
        ],
        paper_guide_evidence_cards=[],
        apply_paper_guide_answer_postprocess=lambda answer, **kwargs: (answer, []),
        maybe_append_library_figure_markdown=lambda answer, **kwargs: answer,
        validate_structured_citations=lambda answer, **kwargs: (answer, {}),
    )

    support_records = out["paper_guide_contracts"]["support_pack"]["support_records"]
    assert len(support_records) == 1
    assert support_records[0]["support_id"] == "slot-figure-1"
    assert support_records[0]["figure_number"] == 3
    assert support_records[0]["panel_letters"] == ["f"]


def test_finalize_generation_answer_contract_snapshot_builds_retrieval_bundle_without_seed():
    out = finalize_runtime._finalize_generation_answer(
        "Core grounded answer.",
        prompt="Explain Figure 3 panel F.",
        prompt_for_user="Explain Figure 3 panel F.",
        answer_hits=[{"meta": {"source_path": "demo.md"}}],
        db_dir="db",
        locked_citation_source=None,
        answer_intent="reading",
        answer_depth="medium",
        answer_output_mode="reading_guide",
        paper_guide_mode=True,
        paper_guide_contract_enabled=False,
        paper_guide_prompt_family="figure_walkthrough",
        paper_guide_special_focus_block="",
        paper_guide_focus_source_path="focus.md",
        paper_guide_direct_source_path="direct.md",
        paper_guide_bound_source_path="bound.md",
        paper_guide_candidate_refs_by_source={"demo.md": [3, 7]},
        paper_guide_support_slots=[],
        paper_guide_evidence_cards=[
            {
                "doc_idx": "1",
                "sid": "s123",
                "source_path": "demo.md",
                "heading": "Results / Figure 3",
                "candidate_refs": ["3", 3],
                "deepread_texts": ["caption line", "caption line", "panel F detail"],
            }
        ],
        apply_paper_guide_answer_postprocess=lambda answer, **kwargs: (answer, []),
        maybe_append_library_figure_markdown=lambda answer, **kwargs: answer,
        validate_structured_citations=lambda answer, **kwargs: (answer, {}),
    )

    retrieval_bundle = out["paper_guide_contracts"]["retrieval_bundle"]
    assert retrieval_bundle["prompt_family"] == "figure_walkthrough"
    assert retrieval_bundle["candidate_refs_by_source"] == {"demo.md": [3, 7]}
    assert retrieval_bundle["evidence_cards"][0]["heading"] == "Results / Figure 3"
    assert retrieval_bundle["evidence_cards"][0]["deepread_texts"] == ["caption line", "panel F detail"]


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
    assert out["paper_guide_contracts"] == {}


def test_finalize_generation_answer_keeps_overview_cites_after_citation_validation_when_family_is_inferred(monkeypatch):
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

    assert "[[CITE:s1234abcd:26]]" not in out["answer"]
    assert "throughput" in out["answer"].lower()
    assert out["answer_quality"]["answer"] == out["answer"]


def test_finalize_generation_answer_injects_minimum_cite_when_missing_after_sanitize_for_citation_lookup(monkeypatch):
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
        locked_citation_source={"sid": "s1234abcd", "source_name": "demo.pdf"},
        answer_intent="reading",
        answer_depth="medium",
        answer_output_mode="reading_guide",
        paper_guide_mode=True,
        paper_guide_contract_enabled=False,
        paper_guide_prompt_family="citation_lookup",
        paper_guide_special_focus_block="",
        paper_guide_focus_source_path="focus.md",
        paper_guide_direct_source_path="direct.md",
        paper_guide_bound_source_path="bound.md",
        paper_guide_candidate_refs_by_source={"demo.md": [26]},
        paper_guide_support_slots=[],
        paper_guide_evidence_cards=[],
        apply_paper_guide_answer_postprocess=lambda answer, **kwargs: (answer, []),
        maybe_append_library_figure_markdown=lambda answer, **kwargs: answer,
        validate_structured_citations=lambda answer, **kwargs: (answer, {"kept": 0}),
    )

    assert "[[CITE:s1234abcd:26]]" not in out["answer"]


def test_finalize_generation_answer_does_not_inject_minimum_cite_for_overview(monkeypatch):
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
        locked_citation_source={"sid": "s1234abcd", "source_name": "demo.pdf"},
        answer_intent="reading",
        answer_depth="medium",
        answer_output_mode="reading_guide",
        paper_guide_mode=True,
        paper_guide_contract_enabled=False,
        paper_guide_prompt_family="overview",
        paper_guide_special_focus_block="",
        paper_guide_focus_source_path="focus.md",
        paper_guide_direct_source_path="direct.md",
        paper_guide_bound_source_path="bound.md",
        paper_guide_candidate_refs_by_source={"demo.md": [26]},
        paper_guide_support_slots=[],
        paper_guide_evidence_cards=[],
        apply_paper_guide_answer_postprocess=lambda answer, **kwargs: (answer, []),
        maybe_append_library_figure_markdown=lambda answer, **kwargs: answer,
        validate_structured_citations=lambda answer, **kwargs: (answer, {"kept": 0}),
    )

    assert "[[CITE:" not in out["answer"]


def test_finalize_generation_answer_skips_supplement_for_cross_paper_query(monkeypatch):
    monkeypatch.setattr(finalize_runtime, "_reconcile_kb_notice", lambda answer, **kwargs: answer)
    monkeypatch.setattr(finalize_runtime, "_apply_answer_contract_v1", lambda answer, **kwargs: answer)
    monkeypatch.setattr(finalize_runtime, "_enhance_kb_miss_fallback", lambda answer, **kwargs: answer)
    monkeypatch.setattr(finalize_runtime, "_build_answer_quality_probe", lambda answer, **kwargs: {"minimum_ok": True, "answer": answer})

    prompt = "Besides this paper, what other papers in my library discuss Fourier single-pixel imaging?"
    out = finalize_runtime._finalize_generation_answer(
        "Only one additional paper appears in the retrieved context.",
        prompt=prompt,
        prompt_for_user=prompt,
        answer_hits=[{"meta": {"source_path": "demo.md"}}],
        db_dir="db",
        locked_citation_source=None,
        answer_intent="reading",
        answer_depth="medium",
        answer_output_mode="reading_guide",
        paper_guide_mode=True,
        paper_guide_contract_enabled=False,
        paper_guide_prompt_family="overview",
        paper_guide_special_focus_block="",
        paper_guide_focus_source_path="focus.md",
        paper_guide_direct_source_path="direct.md",
        paper_guide_bound_source_path="bound.md",
        paper_guide_candidate_refs_by_source={"demo.md": [19]},
        paper_guide_support_slots=[],
        paper_guide_evidence_cards=[],
        paper_guide_retrieval_confidence_hint={"low_confidence": True, "low_confidence_reason": "strict_family_sparse_hits"},
        apply_paper_guide_answer_postprocess=lambda answer, **kwargs: (answer, []),
        maybe_append_library_figure_markdown=lambda answer, **kwargs: answer,
        validate_structured_citations=lambda answer, **kwargs: (answer, {}),
    )

    assert "Supplementary note" not in out["answer"]
    assert "补充说明" not in out["answer"]


def test_finalize_generation_answer_skips_supplement_for_structured_answer(monkeypatch):
    monkeypatch.setattr(finalize_runtime, "_reconcile_kb_notice", lambda answer, **kwargs: answer)
    monkeypatch.setattr(finalize_runtime, "_apply_answer_contract_v1", lambda answer, **kwargs: answer)
    monkeypatch.setattr(finalize_runtime, "_enhance_kb_miss_fallback", lambda answer, **kwargs: answer)
    monkeypatch.setattr(finalize_runtime, "_build_answer_quality_probe", lambda answer, **kwargs: {"minimum_ok": True, "answer": answer})

    structured = "Conclusion: Core answer.\n\nEvidence:\n1. Narrow grounded snippet.\n\nNext Steps:\n1. Verify the cited section."
    out = finalize_runtime._finalize_generation_answer(
        structured,
        prompt="What does the method claim?",
        prompt_for_user="What does the method claim?",
        answer_hits=[{"meta": {"source_path": "demo.md"}}],
        db_dir="db",
        locked_citation_source=None,
        answer_intent="reading",
        answer_depth="medium",
        answer_output_mode="reading_guide",
        paper_guide_mode=True,
        paper_guide_contract_enabled=False,
        paper_guide_prompt_family="overview",
        paper_guide_special_focus_block="",
        paper_guide_focus_source_path="focus.md",
        paper_guide_direct_source_path="direct.md",
        paper_guide_bound_source_path="bound.md",
        paper_guide_candidate_refs_by_source={"demo.md": [26]},
        paper_guide_support_slots=[],
        paper_guide_evidence_cards=[],
        paper_guide_retrieval_confidence_hint={"low_confidence": True, "low_confidence_reason": "strict_family_sparse_hits"},
        apply_paper_guide_answer_postprocess=lambda answer, **kwargs: (answer, []),
        maybe_append_library_figure_markdown=lambda answer, **kwargs: answer,
        validate_structured_citations=lambda answer, **kwargs: (answer, {}),
    )

    assert "Supplementary note" not in out["answer"]
    assert "补充说明" not in out["answer"]
    assert "Conclusion: Core answer." in out["answer"]


def test_finalize_generation_answer_strips_cite_tokens_for_non_citation_answer(monkeypatch):
    monkeypatch.setattr(finalize_runtime, "_reconcile_kb_notice", lambda answer, **kwargs: answer)
    monkeypatch.setattr(finalize_runtime, "_apply_answer_contract_v1", lambda answer, **kwargs: answer)
    monkeypatch.setattr(finalize_runtime, "_enhance_kb_miss_fallback", lambda answer, **kwargs: answer)
    monkeypatch.setattr(finalize_runtime, "_build_answer_quality_probe", lambda answer, **kwargs: {"minimum_ok": True, "answer": answer})

    out = finalize_runtime._finalize_generation_answer(
        "OE-2017 paper [[CITE:s1234abcd:2]].\nSection 2.2 compares the two methods [2].",
        prompt="Which paper in my library directly compares Hadamard single-pixel imaging and Fourier single-pixel imaging?",
        prompt_for_user="Which paper in my library directly compares Hadamard single-pixel imaging and Fourier single-pixel imaging?",
        answer_hits=[{"meta": {"source_path": "demo.md"}}],
        db_dir="db",
        locked_citation_source=None,
        answer_intent="reading",
        answer_depth="medium",
        answer_output_mode="reading_guide",
        paper_guide_mode=False,
        paper_guide_contract_enabled=False,
        paper_guide_prompt_family="",
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

    assert "[[CITE:" not in out["answer"]
    assert "[2]" not in out["answer"]
    assert "Section 2.2 compares the two methods" in out["answer"]


def test_finalize_generation_answer_preserves_numeric_refs_for_citation_lookup(monkeypatch):
    monkeypatch.setattr(finalize_runtime, "_reconcile_kb_notice", lambda answer, **kwargs: answer)
    monkeypatch.setattr(finalize_runtime, "_apply_answer_contract_v1", lambda answer, **kwargs: answer)
    monkeypatch.setattr(finalize_runtime, "_enhance_kb_miss_fallback", lambda answer, **kwargs: answer)
    monkeypatch.setattr(finalize_runtime, "_build_answer_quality_probe", lambda answer, **kwargs: {"minimum_ok": True, "answer": answer})

    out = finalize_runtime._finalize_generation_answer(
        "APR is attributed to prior work [35]. [[CITE:s1234abcd:35]]",
        prompt="Which prior work is RVT attributed to in this paper, and what in-paper citation do they use when introducing it?",
        prompt_for_user="Which prior work is RVT attributed to in this paper, and what in-paper citation do they use when introducing it?",
        answer_hits=[{"meta": {"source_path": "demo.md"}}],
        db_dir="db",
        locked_citation_source={"sid": "s1234abcd", "source_name": "demo.pdf"},
        answer_intent="reading",
        answer_depth="medium",
        answer_output_mode="reading_guide",
        paper_guide_mode=True,
        paper_guide_contract_enabled=False,
        paper_guide_prompt_family="citation_lookup",
        paper_guide_special_focus_block="",
        paper_guide_focus_source_path="focus.md",
        paper_guide_direct_source_path="direct.md",
        paper_guide_bound_source_path="bound.md",
        paper_guide_candidate_refs_by_source={"demo.md": [35]},
        paper_guide_support_slots=[],
        paper_guide_evidence_cards=[],
        apply_paper_guide_answer_postprocess=lambda answer, **kwargs: (answer, []),
        maybe_append_library_figure_markdown=lambda answer, **kwargs: answer,
        validate_structured_citations=lambda answer, **kwargs: (answer, {"kept": 1}),
    )

    assert "[[CITE:" not in out["answer"]
    assert "[35]" in out["answer"]


def test_finalize_generation_answer_prepends_low_confidence_notice_for_paper_guide():
    out = finalize_runtime._finalize_generation_answer(
        "Core claim from the retrieved evidence.",
        prompt="What does the method claim?",
        prompt_for_user="What does the method claim?",
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
            "low_confidence_reason": "strict_family_weak_overlap",
        },
        apply_paper_guide_answer_postprocess=lambda answer, **kwargs: (answer, []),
        maybe_append_library_figure_markdown=lambda answer, **kwargs: answer,
        validate_structured_citations=lambda answer, **kwargs: (answer, {}),
    )

    assert "lower-confidence evidence matching" in out["answer"]
    assert "Core claim from the retrieved evidence." in out["answer"]
    assert out["answer_quality"]["retrieval_confidence"]["low_confidence"] is True


def test_finalize_generation_answer_low_confidence_notice_off_when_hint_absent():
    out = finalize_runtime._finalize_generation_answer(
        "Core claim from the retrieved evidence.",
        prompt="What does the method claim?",
        prompt_for_user="What does the method claim?",
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
    )

    assert out["answer"] == "Core claim from the retrieved evidence."
    assert out["answer_quality"]["retrieval_confidence"] == {}


def test_finalize_generation_answer_low_confidence_notice_includes_candidate_refs():
    out = finalize_runtime._finalize_generation_answer(
        "Core claim from the retrieved evidence.",
        prompt="Which prior work is cited for this method?",
        prompt_for_user="Which prior work is cited for this method?",
        answer_hits=[{"meta": {"source_path": "demo.md"}}],
        db_dir="db",
        locked_citation_source=None,
        answer_intent="reading",
        answer_depth="medium",
        answer_output_mode="reading_guide",
        paper_guide_mode=True,
        paper_guide_contract_enabled=False,
        paper_guide_prompt_family="citation_lookup",
        paper_guide_special_focus_block="",
        paper_guide_focus_source_path="focus.md",
        paper_guide_direct_source_path="direct.md",
        paper_guide_bound_source_path="bound.md",
        paper_guide_candidate_refs_by_source={"demo.md": [4, 22]},
        paper_guide_support_slots=[],
        paper_guide_evidence_cards=[],
        paper_guide_retrieval_confidence_hint={
            "low_confidence": True,
            "low_confidence_reason": "strict_family_sparse_hits",
        },
        apply_paper_guide_answer_postprocess=lambda answer, **kwargs: (
            answer,
            [
                {
                    "resolved_ref_num": 15,
                    "candidate_refs": [4, 15],
                    "support_ref_candidates": [9],
                },
            ],
        ),
        maybe_append_library_figure_markdown=lambda answer, **kwargs: answer,
        validate_structured_citations=lambda answer, **kwargs: (answer, {}),
    )

    assert "Candidate refs for cross-check:" in out["answer"]
    assert "[15], [4], [9], [22]" in out["answer"]
    assert out["answer_quality"]["retrieval_confidence"]["candidate_refs_for_notice"] == [15, 4, 9, 22]


def test_finalize_generation_answer_low_confidence_notice_includes_candidate_refs_zh():
    out = finalize_runtime._finalize_generation_answer(
        "这是当前命中的核心结论。",
        prompt="这个方法引用了哪些工作？",
        prompt_for_user="这个方法引用了哪些工作？请给出处。",
        answer_hits=[{"meta": {"source_path": "demo.md"}}],
        db_dir="db",
        locked_citation_source=None,
        answer_intent="reading",
        answer_depth="medium",
        answer_output_mode="reading_guide",
        paper_guide_mode=True,
        paper_guide_contract_enabled=False,
        paper_guide_prompt_family="citation_lookup",
        paper_guide_special_focus_block="",
        paper_guide_focus_source_path="focus.md",
        paper_guide_direct_source_path="direct.md",
        paper_guide_bound_source_path="bound.md",
        paper_guide_candidate_refs_by_source={},
        paper_guide_support_slots=[],
        paper_guide_evidence_cards=[],
        paper_guide_retrieval_confidence_hint={
            "low_confidence": True,
            "low_confidence_reason": "target_miss",
        },
        apply_paper_guide_answer_postprocess=lambda answer, **kwargs: (
            answer,
            [{"candidate_refs": [7]}],
        ),
        maybe_append_library_figure_markdown=lambda answer, **kwargs: answer,
        validate_structured_citations=lambda answer, **kwargs: (answer, {}),
    )

    assert "低置信证据匹配" in out["answer"]
    assert "候选参考文献：" in out["answer"]
    assert "[7]" in out["answer"]
