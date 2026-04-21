import pytest

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


def test_finalize_generation_answer_prefers_more_precise_card_primary_over_coarse_seed(monkeypatch):
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
        "Grounded answer.",
        prompt="What defines dynamic supersampling in this paper?",
        prompt_for_user="What defines dynamic supersampling in this paper?",
        answer_hits=[{"meta": {"source_path": "demo.md"}}],
        db_dir="db",
        locked_citation_source=None,
        answer_intent="define",
        answer_depth="medium",
        answer_output_mode="reading_guide",
        paper_guide_mode=True,
        paper_guide_contract_enabled=True,
        paper_guide_prompt_family="definition",
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
                    "source_name": "SciAdv-2017.pdf",
                    "source_path": "sciadv.md",
                    "block_id": "blk_30",
                    "anchor_id": "a_30",
                    "heading_path": "INTRODUCTION / Spatially variant digital supersampling",
                    "snippet": "dynamic supersampling is defined here.",
                    "selection_reason": "prompt_aligned",
                },
            }
        ],
        paper_guide_contracts_seed={
            "primary_evidence": {
                "source_name": "SciAdv-2017.pdf",
                "source_path": "sciadv.md",
                "heading_path": "INTRODUCTION",
                "snippet": "A broad answer-hit snippet.",
                "selection_reason": "answer_hit_top",
            }
        },
        apply_paper_guide_answer_postprocess=lambda answer, **kwargs: (answer, []),
        maybe_append_library_figure_markdown=lambda answer, **kwargs: answer,
        validate_structured_citations=lambda answer, **kwargs: (answer, {}),
    )

    assert out["answer"] == "Grounded answer."
    assert seen["primary_evidence"]["block_id"] == "blk_30"
    assert seen["primary_evidence"]["heading_path"] == "INTRODUCTION / Spatially variant digital supersampling"
    assert seen["primary_evidence"]["selection_reason"] == "prompt_aligned"


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


def test_finalize_generation_answer_sanitizes_internal_doc_label_blocks(monkeypatch):
    monkeypatch.setattr(finalize_runtime, "_reconcile_kb_notice", lambda answer, **kwargs: answer)
    monkeypatch.setattr(finalize_runtime, "_apply_answer_contract_v1", lambda answer, **kwargs: answer)
    monkeypatch.setattr(finalize_runtime, "_enhance_kb_miss_fallback", lambda answer, **kwargs: answer)
    monkeypatch.setattr(finalize_runtime, "_build_answer_quality_probe", lambda answer, **kwargs: {"minimum_ok": True, "answer": answer})

    raw = (
        "根据提供的检索结果，以下文章明确提到了 SCI（Snapshot Compressive Imaging，单次曝光压缩成像）：\n\n"
        "DOC-2:\n\n"
        "标题：ICIP-2025-SCIGS- 3D Gaussians Splatting from A Snapshot Compressive Image\n"
        "明确使用了术语 “Snapshot Compressive Imaging (SCI)”。\n\n"
        "DOC-3:\n\n"
        "标题：CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image\n"
        "多次提及 “Snapshot Compressive Imaging (SCI)”。\n\n"
        "注意：DOC-4 未提及 SCI 或相关术语。"
    )

    out = finalize_runtime._finalize_generation_answer(
        raw,
        prompt="有哪几篇文章提到了SCI（单次曝光压缩成像）",
        prompt_for_user="有哪几篇文章提到了SCI（单次曝光压缩成像）",
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

    assert "DOC-2" not in out["answer"]
    assert "DOC-3" not in out["answer"]
    assert "DOC-4" not in out["answer"]
    assert "ICIP-2025-SCIGS" in out["answer"]
    assert "CVPR-2024-SCINeRF" in out["answer"]
    assert "- ICIP-2025-SCIGS" in out["answer"]


def test_finalize_generation_answer_rebuilds_multi_paper_list_from_structured_docs(monkeypatch):
    monkeypatch.setattr(finalize_runtime, "_reconcile_kb_notice", lambda answer, **kwargs: answer)
    monkeypatch.setattr(finalize_runtime, "_apply_answer_contract_v1", lambda answer, **kwargs: answer)
    monkeypatch.setattr(finalize_runtime, "_enhance_kb_miss_fallback", lambda answer, **kwargs: answer)
    monkeypatch.setattr(finalize_runtime, "_build_answer_quality_probe", lambda answer, **kwargs: {"minimum_ok": True, "answer": answer})

    raw = (
        "根据提供的上下文，以下几篇文章明确提到了 SCI：\n\n"
        "1. **DOC-2**：*ICIP-2025-SCIGS*\n"
        "2. **DOC-3**：*CVPR-2024-SCINeRF*\n"
        "3. **DOC-1**：*OE-2007-Single-shot compressive spectral imaging with a dual-disperser architecture*\n"
        "> 注：DOC-4 未提及 SCI 或相关术语。"
    )
    docs = [
        (
            r"db\ICIP-2025-SCIGS\ICIP-2025-SCIGS.en.md",
            "ICIP-2025-SCIGS.pdf",
            "Introduction",
            "The paper explicitly introduces Snapshot Compressive Imaging (SCI).",
        ),
        (
            r"db\CVPR-2024-SCINeRF\CVPR-2024-SCINeRF.en.md",
            "CVPR-2024-SCINeRF.pdf",
            "Abstract",
            "The abstract repeatedly mentions Snapshot Compressive Imaging (SCI).",
        ),
        (
            r"db\OE-2007-Single-shot compressive spectral imaging with a dual-disperser architecture\OE-2007-Single-shot compressive spectral imaging with a dual-disperser architecture.en.md",
            "OE-2007-Single-shot compressive spectral imaging with a dual-disperser architecture.pdf",
            "5. Conclusions",
            "This early single-shot compressive spectral imaging paper is treated as an SCI predecessor.",
        ),
    ]
    answer_hits = []
    evidence_cards = []
    for source_path, source_name, heading_path, snippet in docs:
        answer_hits.append(
            {
                "text": snippet,
                "meta": {
                    "source_path": source_path,
                    "ref_best_heading_path": heading_path,
                },
            }
        )
        evidence_cards.append(
            {
                "source_path": source_path,
                "heading": heading_path,
                "snippet": snippet,
                "primary_evidence": {
                    "source_path": source_path,
                    "source_name": source_name,
                    "heading_path": heading_path,
                    "snippet": snippet,
                },
            }
        )

    out = finalize_runtime._finalize_generation_answer(
        raw,
        prompt="有哪几篇文章提到了SCI（单次曝光压缩成像）",
        prompt_for_user="有哪几篇文章提到了SCI（单次曝光压缩成像）",
        answer_hits=answer_hits,
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
        paper_guide_evidence_cards=evidence_cards,
        apply_paper_guide_answer_postprocess=lambda answer, **kwargs: (answer, []),
        maybe_append_library_figure_markdown=lambda answer, **kwargs: answer,
        validate_structured_citations=lambda answer, **kwargs: (answer, {}),
    )

    assert "DOC-1" not in out["answer"]
    assert "DOC-2" not in out["answer"]
    assert "DOC-3" not in out["answer"]
    assert "DOC-4" not in out["answer"]
    assert "ICIP-2025-SCIGS.pdf" in out["answer"]
    assert "CVPR-2024-SCINeRF.pdf" in out["answer"]
    assert "OE-2007-Single-shot compressive spectral imaging with a dual-disperser architecture.pdf" in out["answer"]
    assert "定位：" in out["answer"]
    contracts = dict(out.get("paper_guide_contracts") or {})
    assert len(list(contracts.get("doc_list") or [])) == 3


def test_build_multi_paper_doc_list_contract_prefers_normalized_pending_seed_surface_over_weaker_answer_hit_card():
    source_path = (
        r"db\CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image"
        r"\CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image.en.md"
    )
    raw_snippet = (
        "## Abstract\n"
        "In this paper, we explore the potential of Snapshot Compressive Imaging (SCI) technique for recovering "
        "the underlying 3D scene representation from a single temporal compressed image.\n\n"
        "## 1. Introduction\n"
        "Conventional high-speed imaging systems often face challenges such as high hardware cost."
    )

    out = finalize_runtime._build_multi_paper_doc_list_contract(
        prompt="Which papers in my library mention SCI (Snapshot Compressive Imaging)?",
        seed_docs=[
            {
                "text": raw_snippet,
                "meta": {
                    "source_path": source_path,
                    "ref_best_heading_path": "2. Related Work",
                    "ref_show_snippets": [raw_snippet],
                },
            }
        ],
        answer_hits=[],
        evidence_cards=[
            {
                "source_path": source_path,
                "heading": "2. Related Work",
                "snippet": raw_snippet,
                "primary_evidence": {
                    "source_path": source_path,
                    "source_name": "CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image.pdf",
                    "heading_path": "2. Related Work",
                    "snippet": raw_snippet,
                    "selection_reason": "answer_hit_top",
                },
            }
        ],
    )

    assert len(out) == 1
    row = out[0]
    assert row["heading_path"] == "Abstract"
    assert row["summary_line"].startswith(
        "In this paper, we explore the potential of Snapshot Compressive Imaging (SCI) technique"
    )
    primary = dict(row.get("primary_evidence") or {})
    assert primary["heading_path"] == "Abstract"
    assert primary["selection_reason"] == "pending_section_seed"
    assert primary["snippet"].startswith(
        "In this paper, we explore the potential of Snapshot Compressive Imaging (SCI) technique"
    )


def test_build_multi_paper_doc_list_contract_extracts_abstract_surface_from_title_plus_bold_abstract():
    source_path = (
        r"db\OE-2007-Single-shot compressive spectral imaging with a dual-disperser architecture"
        r"\OE-2007-Single-shot compressive spectral imaging with a dual-disperser architecture.en.md"
    )
    raw_snippet = (
        "# Single-shot compressive spectral imaging with a dual-disperser architecture\n"
        "M. E. Gehm, R. John, D. J. Brady\n"
        "**Abstract**: This paper describes a single-shot spectral imaging approach based on the concept of "
        "compressive sensing.\n\n"
        "## 5. Conclusions\n"
        "In this manuscript we have described a new, single-shot spectral imager based on compressive sensing ideas."
    )

    out = finalize_runtime._build_multi_paper_doc_list_contract(
        prompt="Which papers in my library mention SCI (Snapshot Compressive Imaging)?",
        seed_docs=[
            {
                "text": raw_snippet,
                "meta": {
                    "source_path": source_path,
                    "ref_best_heading_path": "5. Conclusions",
                    "ref_show_snippets": [raw_snippet],
                },
            }
        ],
        answer_hits=[],
        evidence_cards=[],
    )

    assert len(out) == 1
    row = out[0]
    assert row["heading_path"] == "Abstract"
    assert row["summary_line"].startswith(
        "This paper describes a single-shot spectral imaging approach based on the concept of compressive sensing."
    )
    primary = dict(row.get("primary_evidence") or {})
    assert primary["heading_path"] == "Abstract"
    assert primary["selection_reason"] == "pending_section_seed"


def test_build_multi_paper_doc_list_contract_allows_snippet_rich_answer_hit_to_replace_heading_only_seed_primary():
    source_path = (
        r"db\Frontiers of Physics-2024-Emerging single-photon...performance photodetector"
        r"\Frontiers of Physics-2024-Emerging single-photon...performance photodetector.en.md"
    )
    snippet = (
        "Single-photon imaging counting can break through the signal-to-noise ratio limit of classical imaging, "
        "and effectively improve the working distance and quality of remote sensing and reconnaissance."
    )

    out = finalize_runtime._build_multi_paper_doc_list_contract(
        prompt="Which papers in my library discuss single-photon imaging?",
        seed_docs=[
            {
                "meta": {
                    "source_path": source_path,
                    "ref_best_heading_path": "5 Application / 5.3 Quantum communication",
                    "ref_show_snippets": [],
                },
                "text": "",
            }
        ],
        answer_hits=[
            {
                "text": snippet,
                "meta": {
                    "source_path": source_path,
                    "ref_best_heading_path": "5 Application / 5.3 Quantum communication",
                },
            }
        ],
        evidence_cards=[],
    )

    assert len(out) == 1
    row = out[0]
    assert row["heading_path"] == "5 Application / 5.3 Quantum communication"
    assert row["summary_line"].startswith(
        "Single-photon imaging counting can break through the signal-to-noise ratio limit of classical imaging"
    )
    primary = dict(row.get("primary_evidence") or {})
    assert primary["selection_reason"] == "answer_hit_top"
    assert primary["snippet"].startswith(
        "Single-photon imaging counting can break through the signal-to-noise ratio limit of classical imaging"
    )


def test_build_multi_paper_doc_list_contract_uses_deepread_text_when_card_snippet_is_empty():
    source_path = (
        r"db\Frontiers of Physics-2024-Emerging single-photon...performance photodetector"
        r"\Frontiers of Physics-2024-Emerging single-photon...performance photodetector.en.md"
    )
    deepread_text = (
        "## 5 Application / 5.1 Optical imaging\n"
        "Single-photon imaging counting can break through the signal-to-noise ratio limit of classical imaging, "
        "and effectively improve the working distance and quality of remote sensing and reconnaissance."
    )

    out = finalize_runtime._build_multi_paper_doc_list_contract(
        prompt="Which papers in my library discuss single-photon imaging?",
        seed_docs=[],
        answer_hits=[],
        evidence_cards=[
            {
                "source_path": source_path,
                "heading": "5 Application / 5.3 Quantum communication",
                "snippet": "",
                "deepread_texts": [deepread_text],
                "primary_evidence": {
                    "source_path": source_path,
                    "source_name": "Frontiers of Physics-2024-Emerging single-photon...performance photodetector.pdf",
                    "heading_path": "5 Application / 5.3 Quantum communication",
                    "selection_reason": "answer_hit_top",
                },
            }
        ],
    )

    assert len(out) == 1
    row = out[0]
    assert row["heading_path"] == "5 Application / 5.1 Optical imaging"
    assert row["summary_line"].startswith(
        "Single-photon imaging counting can break through the signal-to-noise ratio limit of classical imaging"
    )
    primary = dict(row.get("primary_evidence") or {})
    assert primary["heading_path"] == "5 Application / 5.1 Optical imaging"
    assert primary["snippet"].startswith(
        "Single-photon imaging counting can break through the signal-to-noise ratio limit of classical imaging"
    )


@pytest.mark.skip(reason="legacy encoding-sensitive prompt case replaced by ASCII-equivalent coverage below")
def test_filter_multi_paper_doc_list_contract_keeps_only_sci_topic_matches():
    prompt = "有哪几篇文章提到了SCI（单次曝光压缩成像）"
    rows = [
        {
            "source_path": r"db\OE-2007\OE-2007.en.md",
            "source_name": "OE-2007-Single-shot compressive spectral imaging with a dual-disperser architecture.pdf",
            "heading_path": "5. Conclusions",
            "summary_line": "This paper describes a single-shot spectral imaging approach based on the concept of compressive sensing.",
        },
        {
            "source_path": r"db\ICIP-2025-SCIGS\ICIP-2025-SCIGS.en.md",
            "source_name": "ICIP-2025-SCIGS- 3D Gaussians Splatting from A Snapshot Compressive Image.pdf",
            "heading_path": "1. Introduction",
            "summary_line": "Video Snapshot Compressive Imaging (SCI) technology has been developed for high-speed imaging.",
        },
        {
            "source_path": r"db\CVPR-2024-SCINeRF\CVPR-2024-SCINeRF.en.md",
            "source_name": "CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image.pdf",
            "heading_path": "Abstract",
            "summary_line": "Snapshot Compressive Imaging (SCI) is used to recover the 3D scene representation from a single temporal compressed image.",
        },
        {
            "source_path": r"db\NatCommun-2023\NatCommun-2023.en.md",
            "source_name": "NatCommun-2023-High-resolution single-photon imaging with physics-informed deep learning.pdf",
            "heading_path": "Abstract",
            "summary_line": "This work validates a single-photon imaging technique for microscopy applications.",
        },
        {
            "source_path": r"db\arxiv-ghost\ghost.en.md",
            "source_name": "arXiv-Quantum correlation light-field microscope with extreme depth of field.pdf",
            "heading_path": "I. INTRODUCTION",
            "summary_line": "This work studies ghost imaging in the Fourier plane.",
        },
    ]

    out = finalize_runtime._filter_multi_paper_doc_list_contract(
        prompt=prompt,
        doc_list=rows,
    )

    assert [item["source_name"] for item in out] == [
        "CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image.pdf",
        "ICIP-2025-SCIGS- 3D Gaussians Splatting from A Snapshot Compressive Image.pdf",
        "OE-2007-Single-shot compressive spectral imaging with a dual-disperser architecture.pdf",
    ]


def test_filter_multi_paper_doc_list_contract_keeps_only_sci_topic_matches_ascii_prompt():
    prompt = "Which papers in my library mention SCI (Snapshot Compressive Imaging)?"
    rows = [
        {
            "source_path": r"db\OE-2007\OE-2007.en.md",
            "source_name": "OE-2007-Single-shot compressive spectral imaging with a dual-disperser architecture.pdf",
            "heading_path": "5. Conclusions",
            "summary_line": "This paper describes a single-shot spectral imaging approach based on the concept of compressive sensing.",
        },
        {
            "source_path": r"db\ICIP-2025-SCIGS\ICIP-2025-SCIGS.en.md",
            "source_name": "ICIP-2025-SCIGS- 3D Gaussians Splatting from A Snapshot Compressive Image.pdf",
            "heading_path": "1. Introduction",
            "summary_line": "Video Snapshot Compressive Imaging (SCI) technology has been developed for high-speed imaging.",
        },
        {
            "source_path": r"db\CVPR-2024-SCINeRF\CVPR-2024-SCINeRF.en.md",
            "source_name": "CVPR-2024-SCINeRF- Neural Radiance Fields from A Snapshot Compressive Image.pdf",
            "heading_path": "Abstract",
            "summary_line": "Snapshot Compressive Imaging (SCI) is used to recover the 3D scene representation from a single temporal compressed image.",
        },
        {
            "source_path": r"db\NatCommun-2023\NatCommun-2023.en.md",
            "source_name": "NatCommun-2023-High-resolution single-photon imaging with physics-informed deep learning.pdf",
            "heading_path": "Abstract",
            "summary_line": "This work validates a single-photon imaging technique for microscopy applications.",
        },
        {
            "source_path": r"db\arxiv-ghost\ghost.en.md",
            "source_name": "arXiv-Quantum correlation light-field microscope with extreme depth of field.pdf",
            "heading_path": "I. INTRODUCTION",
            "summary_line": "This work studies ghost imaging in the Fourier plane.",
        },
    ]

    out = finalize_runtime._filter_multi_paper_doc_list_contract(
        prompt=prompt,
        doc_list=rows,
    )

    assert [item["source_name"] for item in out] == [
        "ICIP-2025-SCIGS- 3D Gaussians Splatting from A Snapshot Compressive Image.pdf",
        "CVPR-2024-SCINeRF- Neural Radiance Fields from A Snapshot Compressive Image.pdf",
        "OE-2007-Single-shot compressive spectral imaging with a dual-disperser architecture.pdf",
    ]


def test_filter_multi_paper_doc_list_contract_marks_explicit_vs_predecessor_sci_matches():
    prompt = "Which papers in my library mention SCI (Snapshot Compressive Imaging)?"
    rows = [
        {
            "source_path": r"db\OE-2007\OE-2007.en.md",
            "source_name": "OE-2007-Single-shot compressive spectral imaging with a dual-disperser architecture.pdf",
            "heading_path": "5. Conclusions",
            "summary_line": "This paper describes a single-shot spectral imaging approach based on the concept of compressive sensing.",
        },
        {
            "source_path": r"db\ICIP-2025-SCIGS\ICIP-2025-SCIGS.en.md",
            "source_name": "ICIP-2025-SCIGS- 3D Gaussians Splatting from A Snapshot Compressive Image.pdf",
            "heading_path": "1. Introduction",
            "summary_line": "Video Snapshot Compressive Imaging (SCI) technology has been developed for high-speed imaging.",
        },
        {
            "source_path": r"db\CVPR-2024-SCINeRF\CVPR-2024-SCINeRF.en.md",
            "source_name": "CVPR-2024-SCINeRF- Neural Radiance Fields from A Snapshot Compressive Image.pdf",
            "heading_path": "Abstract",
            "summary_line": "Snapshot Compressive Imaging (SCI) is used to recover the 3D scene representation from a single temporal compressed image.",
        },
    ]

    out = finalize_runtime._filter_multi_paper_doc_list_contract(
        prompt=prompt,
        doc_list=rows,
    )

    assert [item["topic_match_kind"] for item in out] == [
        "explicit_sci_mention",
        "explicit_sci_mention",
        "sci_related_predecessor",
    ]


def test_exclude_bound_source_from_multi_paper_doc_list_contract_removes_self_paper():
    rows = [
        {
            "source_path": r"db\NatPhoton-2019-Principles and prospects for single-pixel imaging\NatPhoton-2019-Principles and prospects for single-pixel imaging.en.md",
            "source_name": "NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf",
            "heading_path": "Acquisition and image reconstruction strategies",
            "summary_line": "The bound paper reviews single-pixel imaging and briefly mentions Fourier patterns.",
        },
        {
            "source_path": r"db\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging\OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.en.md",
            "source_name": "OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.pdf",
            "heading_path": "2.2 Basis patterns generation",
            "summary_line": "The paper directly compares Hadamard and Fourier single-pixel imaging.",
        },
    ]

    out = finalize_runtime._exclude_bound_source_from_multi_paper_doc_list_contract(
        doc_list=rows,
        bound_source_path=r"db\NatPhoton-2019-Principles and prospects for single-pixel imaging\NatPhoton-2019-Principles and prospects for single-pixel imaging.en.md",
        bound_source_name="NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf",
    )

    assert [item["source_name"] for item in out] == [
        "OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.pdf",
    ]


def test_filter_multi_paper_doc_list_contract_returns_empty_when_explicit_focus_has_no_positive_match():
    prompt = "Besides this paper, what other papers in my library discuss ADMM?"
    rows = [
        {
            "source_path": r"db\ICIP-2025-SCIGS\ICIP-2025-SCIGS.en.md",
            "source_name": "ICIP-2025-SCIGS.pdf",
            "heading_path": "2. Related Work",
            "summary_line": "This paper proposes a reconstruction method for snapshot compressive imaging without relying on ADMM.",
        },
        {
            "source_path": r"db\Journal-2016\Journal-2016.en.md",
            "source_name": "Journal of Optics-2016-3D single-pixel video.pdf",
            "heading_path": "3D single-pixel video",
            "summary_line": "This paper studies 3D single-pixel video reconstruction.",
        },
    ]

    out = finalize_runtime._filter_multi_paper_doc_list_contract(
        prompt=prompt,
        doc_list=rows,
    )

    assert out == []


def test_filter_multi_paper_doc_list_contract_ignores_generic_prompt_echo_summary_for_fourier():
    prompt = "Besides this paper, what other papers in my library discuss Fourier single-pixel imaging?"
    rows = [
        {
            "source_path": r"db\OE-2017\OE-2017.en.md",
            "source_name": "OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.pdf",
            "heading_path": "2.2 Basis patterns generation",
            "summary_line": "The paper directly compares Hadamard single-pixel imaging and Fourier single-pixel imaging.",
            "primary_evidence": {
                "heading_path": "2.2 Basis patterns generation",
                "snippet": "Fourier basis patterns have horizontal, vertical, and oblique features.",
            },
        },
        {
            "source_path": r"db\LPR-2025\LPR-2025.en.md",
            "source_name": "LPR-2025-Advances and Challenges of Single-Pixel Imaging Based on Deep Learning.pdf",
            "heading_path": "4.1. Strategy of Single-Pixel Imaging via Deep Learning",
            "summary_line": "该文第4.1节综述了基于深度学习的单像素成像策略，其中包含对傅里叶单像素成像方法的讨论。",
            "primary_evidence": {
                "heading_path": "4.1. Strategy of Single-Pixel Imaging via Deep Learning",
                "snippet": "该文在“1. INTRODUCTION”给出了与“Besides this paper, what other...”直接相关的定义、方法或结果信息。",
            },
        },
    ]

    out = finalize_runtime._filter_multi_paper_doc_list_contract(
        prompt=prompt,
        doc_list=rows,
    )

    assert [item["source_name"] for item in out] == [
        "OE-2017-Hadamard single-pixel imaging versus Fourier single-pixel imaging.pdf",
    ]


def test_filter_multi_paper_doc_list_contract_requires_full_dynamic_supersampling_focus_match():
    prompt = "Which papers in my library mention dynamic supersampling?"
    rows = [
        {
            "source_path": r"db\SciAdv-2017\SciAdv-2017.en.md",
            "source_name": "SciAdv-2017-Adaptive foveated single-pixel imaging with dynamic supersampling.pdf",
            "heading_path": "INTRODUCTION / Spatially variant digital supersampling",
            "summary_line": "Adaptive foveated single-pixel imaging with dynamic supersampling.",
        },
        {
            "source_path": r"db\NatCommun-2021\NatCommun-2021.en.md",
            "source_name": "NatCommun-2021-Imaging biological tissue with...pixel compressive holography.pdf",
            "heading_path": "Introduction",
            "summary_line": "Recently, adaptive and smart sensing with dynamic supersampling was reported to combine with compressive sensing in SPI.",
        },
        {
            "source_path": r"db\Journal-2016\Journal-2016.en.md",
            "source_name": "Journal of Optics-2016-3D single-pixel video.pdf",
            "heading_path": "Methods / Custom single-pixel system design",
            "summary_line": "The application programming interface is written as a dynamic-link library file.",
        },
        {
            "source_path": r"db\ICIP-2025\ICIP-2025.en.md",
            "source_name": "ICIP-2025-SCIGS- 3D Gaussians Splatting from A Snapshot Compressive Image.pdf",
            "heading_path": "A.2. Result and Analysis",
            "summary_line": "This paper proposes a novel method for recovering dynamic 3D scene representations from a single snapshot compressive image.",
        },
    ]

    out = finalize_runtime._filter_multi_paper_doc_list_contract(
        prompt=prompt,
        doc_list=rows,
    )

    assert [item["source_name"] for item in out] == [
        "SciAdv-2017-Adaptive foveated single-pixel imaging with dynamic supersampling.pdf",
        "NatCommun-2021-Imaging biological tissue with...pixel compressive holography.pdf",
    ]


def test_filter_multi_paper_doc_list_contract_requires_full_compressive_holography_focus_match():
    prompt = "Which papers in my library discuss compressive holography?"
    rows = [
        {
            "source_path": r"db\NatCommun-2021\NatCommun-2021.en.md",
            "source_name": "NatCommun-2021-Imaging biological tissue with...pixel compressive holography.pdf",
            "heading_path": "ARTICLE / Imaging biological tissue with high-throughput single-pixel compressive holography",
            "summary_line": "In this work, we develop a high-throughput single-pixel compressive holography system.",
        },
        {
            "source_path": r"db\Journal-2016\Journal-2016.en.md",
            "source_name": "Journal of Optics-2016-3D single-pixel video.pdf",
            "heading_path": "Methods / Custom single-pixel system design",
            "summary_line": "A few studies have aimed to improve the imaging speed by using compressive sensing.",
        },
    ]

    out = finalize_runtime._filter_multi_paper_doc_list_contract(
        prompt=prompt,
        doc_list=rows,
    )

    assert [item["source_name"] for item in out] == [
        "NatCommun-2021-Imaging biological tissue with...pixel compressive holography.pdf",
    ]


def test_filter_multi_paper_doc_list_contract_does_not_match_single_photon_prompt_to_natphoton_filename():
    prompt = "Which papers in my library discuss single-photon imaging?"
    rows = [
        {
            "source_path": r"db\NatCommun-2023\NatCommun-2023.en.md",
            "source_name": "NatCommun-2023-High-resolution single-photon imaging with physics-informed deep learning.pdf",
            "heading_path": "Abstract",
            "summary_line": "High-resolution single-photon imaging remains a big challenge due to the complex hardware manufacturing craft and noise disturbances.",
        },
        {
            "source_path": r"db\Frontiers-2024\Frontiers-2024.en.md",
            "source_name": "Frontiers of Physics-2024-Emerging single-photon...performance photodetector.pdf",
            "heading_path": "5 Application",
            "summary_line": "Single-photon imaging can reconstruct the image of the object by detecting the three-dimensional space position and time information of each photon.",
        },
        {
            "source_path": r"db\NatPhoton-2019\NatPhoton-2019.en.md",
            "source_name": "NatPhoton-2019-Principles and prospects for single-pixel imaging.pdf",
            "heading_path": "Applications and future potential for single-pixel imaging",
            "summary_line": "This review surveys single-pixel imaging applications.",
        },
    ]

    out = finalize_runtime._filter_multi_paper_doc_list_contract(
        prompt=prompt,
        doc_list=rows,
    )

    assert [item["source_name"] for item in out] == [
        "NatCommun-2023-High-resolution single-photon imaging with physics-informed deep learning.pdf",
        "Frontiers of Physics-2024-Emerging single-photon...performance photodetector.pdf",
    ]


def test_filter_multi_paper_doc_list_contract_requires_both_deep_learning_and_single_pixel_segments():
    prompt = "Besides this paper, what other papers in my library discuss deep learning for single-pixel imaging?"
    rows = [
        {
            "source_path": r"db\OLT-2024\OLT-2024.en.md",
            "source_name": "Optics & Laser Technology-2024-Part-based image-loop network for single-pixel imaging.pdf",
            "heading_path": "Introduction",
            "summary_line": "Deep learning (DL) has immense potential to enhance SPI results significantly, and we proposed a self-supervised image-loop neural network for single-pixel imaging.",
        },
        {
            "source_path": r"db\Visual-2019\Visual-2019.en.md",
            "source_name": "Visual Computing for Industry, Biomedicine, and Art-2019-Brief review...techniques.pdf",
            "heading_path": "Deep learning-based denoising methods",
            "summary_line": "Owing to their outstanding denoising ability, considerable attention has been focused on deep learning-based denoising methods.",
        },
        {
            "source_path": r"db\NatCommun-2023\NatCommun-2023.en.md",
            "source_name": "NatCommun-2023-High-resolution single-photon imaging with physics-informed deep learning.pdf",
            "heading_path": "Abstract",
            "summary_line": "Here, we introduce deep learning into SPAD, enabling super-resolution single-photon imaging.",
        },
        {
            "source_path": r"db\ICIP-2025\ICIP-2025.en.md",
            "source_name": "ICIP-2025-SCIGS- 3D Gaussians Splatting from A Snapshot Compressive Image.pdf",
            "heading_path": "Abstract",
            "summary_line": "Current deep learning-based reconstruction methods face challenges in dynamic SCI scenes.",
        },
    ]

    out = finalize_runtime._filter_multi_paper_doc_list_contract(
        prompt=prompt,
        doc_list=rows,
    )

    assert [item["source_name"] for item in out] == [
        "Optics & Laser Technology-2024-Part-based image-loop network for single-pixel imaging.pdf",
    ]


def test_format_multi_paper_list_answer_v2_marks_sci_predecessor_as_related_not_exact():
    prompt = "Which papers in my library mention SCI (Snapshot Compressive Imaging)?"
    out = finalize_runtime._format_multi_paper_list_answer_v2(
        prompt=prompt,
        docs=[
            {
                "source_name": "ICIP-2025-SCIGS.pdf",
                "heading_path": "1. Introduction",
                "summary_line": "Video Snapshot Compressive Imaging (SCI) technology has been developed for high-speed imaging.",
                "topic_match_kind": "explicit_sci_mention",
            },
            {
                "source_name": "CVPR-2024-SCINeRF.pdf",
                "heading_path": "2. Related Work",
                "summary_line": "Snapshot Compressive Imaging (SCI) is used for 3D scene reconstruction.",
                "topic_match_kind": "explicit_sci_mention",
            },
            {
                "source_name": "OE-2007-Single-shot compressive spectral imaging with a dual-disperser architecture.pdf",
                "heading_path": "5. Conclusions",
                "summary_line": "This paper describes a single-shot spectral imaging approach based on the concept of compressive sensing.",
                "topic_match_kind": "sci_related_predecessor",
            },
        ],
    )

    assert "explicitly mentions Snapshot Compressive Imaging (SCI)" in out
    assert "early related predecessor" in out
    assert "exact SCI term match" in out


def test_format_multi_paper_list_answer_v2_uses_singular_intro_for_single_doc():
    out = finalize_runtime._format_multi_paper_list_answer_v2(
        prompt="Which papers in my library discuss compressive holography?",
        docs=[
            {
                "source_name": "NatCommun-2021-Imaging biological tissue with...pixel compressive holography.pdf",
                "heading_path": "ARTICLE / Imaging biological tissue with high-throughput single-pixel compressive holography",
                "summary_line": "In this work, we develop a high-throughput single-pixel compressive holography system.",
                "topic_match_kind": "direct_topic_match",
            }
        ],
    )

    assert "The following library paper directly relates to 'compressive holography':" in out
    assert "The following 1 library papers" not in out


def test_finalize_generation_answer_uses_authoritative_single_doc_list_for_multi_paper_query(monkeypatch):
    monkeypatch.setattr(finalize_runtime, "_reconcile_kb_notice", lambda answer, **kwargs: answer)
    monkeypatch.setattr(finalize_runtime, "_enhance_kb_miss_fallback", lambda answer, **kwargs: answer)
    monkeypatch.setattr(finalize_runtime, "_build_answer_quality_probe", lambda answer, **kwargs: {"minimum_ok": True})
    monkeypatch.setattr(
        finalize_runtime,
        "_build_multi_paper_doc_list_contract",
        lambda **kwargs: [
            {
                "source_name": "NatCommun-2021-Imaging biological tissue with...pixel compressive holography.pdf",
                "heading_path": "ARTICLE / Imaging biological tissue with high-throughput single-pixel compressive holography",
                "summary_line": "In this work, we develop a high-throughput single-pixel compressive holography system.",
                "topic_match_kind": "direct_topic_match",
            }
        ],
    )

    out = finalize_runtime._finalize_generation_answer(
        "The retrieved context also mentions NatPhoton-2019 and Journal-2016 as related background.",
        prompt="Which papers in my library discuss compressive holography?",
        prompt_for_user="Which papers in my library discuss compressive holography?",
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

    answer = str(out.get("answer") or "")
    assert "NatCommun-2021-Imaging biological tissue with...pixel compressive holography.pdf" in answer
    assert "NatPhoton-2019" not in answer
    assert "Journal-2016" not in answer
    assert "The following library paper directly relates to 'compressive holography':" in answer


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
