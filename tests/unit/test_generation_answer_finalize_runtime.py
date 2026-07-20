from pathlib import Path

import pytest

import kb.generation_answer_finalize_runtime as finalize_runtime


def test_origin_question_requests_upstream_citation_lookup() -> None:
    assert finalize_runtime._prompt_explicitly_requests_citation_lookup(
        "ADMM 是作者自己发明的吗？我应该把它当成这篇论文的新东西吗？"
    )


def test_answer_source_request_does_not_request_upstream_citation_lookup() -> None:
    assert not finalize_runtime._prompt_explicitly_requests_citation_lookup(
        "这篇论文建模了哪些真实退化？请只根据本文用三点回答，并给出对应引用。"
    )


def test_english_answer_source_request_does_not_request_upstream_citation_lookup() -> None:
    assert not finalize_runtime._prompt_explicitly_requests_citation_lookup(
        "What degradations are modeled? Give corresponding citations for each claim."
    )


def test_answer_audit_doc_labels_become_user_facing_source_labels() -> None:
    out = finalize_runtime._replace_answer_audit_doc_labels(
        "来源核对：DOC-2 的正文与标题一致 [10002]。"
    )

    assert out == "来源核对：来源 [2] 的正文与标题一致 [10002]。"
    assert "DOC-" not in out


def test_answer_audit_strips_internal_citation_format_review_unless_requested() -> None:
    answer = (
        "## 审查结果\n\n四篇论文标题与依据一致。\n\n"
        "### 3. 引用编号问题\n\n模型没有使用 [10001] 偏移标记。\n\n"
        "### 4. 总结\n\n核心路线准确；二是引用编号格式不符合要求。"
    )

    out = finalize_runtime._strip_answer_audit_internal_citation_review(
        answer,
        prompt="Audit the previous answer and verify its source bindings.",
    )

    assert "四篇论文标题与依据一致" in out
    assert "### 4. 总结" in out
    assert "引用编号" not in out
    assert "10001" not in out

    preserved = finalize_runtime._strip_answer_audit_internal_citation_review(
        answer,
        prompt="Audit the previous answer's citation format and marker numbering.",
    )
    assert "10001" in preserved


def test_canceled_generation_answer_keeps_prose_and_hides_internal_markers() -> None:
    out = finalize_runtime._sanitize_canceled_generation_answer(
        "## Partial answer\n\nDOC-1 supports the first claim [10001]. "
        "DOC-3 supports the comparison [10003]. [[SUPPORT:DOC-1]]",
        prompt="Compare the selected papers.",
        has_hits=True,
    )

    assert "Partial answer" in out
    assert "first claim [1]" in out
    assert "comparison [3]" in out
    assert "10001" not in out
    assert "10003" not in out
    assert "DOC-" not in out
    assert "SUPPORT" not in out
    assert out.endswith("(Generation canceled)")


def test_canceled_generation_answer_without_partial_is_stable() -> None:
    assert finalize_runtime._sanitize_canceled_generation_answer("") == "(Generation canceled)"


def test_numeric_citation_normalization_collapses_provider_double_brackets_and_separators() -> None:
    assert finalize_runtime._normalize_double_numeric_citations(
        "Evidence [[4]], [[5；2]], and [[3、1]]."
    ) == "Evidence [4], [5；2], and [3、1]."


def test_offset_citations_inside_double_brackets_become_public_markers() -> None:
    converted = finalize_runtime._strip_citation_offset("Evidence [[10004;10005]].")
    assert finalize_runtime._normalize_double_numeric_citations(converted) == "Evidence [4,5]."


def test_stripped_structured_citation_does_not_leave_empty_bracket_shell() -> None:
    out = finalize_runtime._strip_final_answer_citation_markers(
        "Claim [ [[CITE:source:12]] ].",
        preserve_numeric_markers=True,
        preserve_structured_markers=False,
    )

    assert out == "Claim."
    assert finalize_runtime._sanitize_empty_markdown_label_fragments("- [ ] task") == "- [ ] task"


def test_retrieval_window_does_not_masquerade_as_whole_library() -> None:
    out = finalize_runtime._normalize_retrieval_window_claims(
        "根据您提供的库中文献（共2篇），没有任何一篇文献涉及该主题。"
        "结论：库中文献资源不足以支撑这个问题。",
        prompt="请结合库中文献回答。",
    )

    assert "共2篇" not in out
    assert "本轮检索到的候选文献" in out
    assert "本轮检索证据不足" in out
    assert "库中文献资源不足" not in out


def test_explicit_library_inventory_count_requires_verified_contract() -> None:
    answer = "库中文献（共42篇）。"
    assert finalize_runtime._normalize_retrieval_window_claims(
        answer,
        prompt="我的文献库里有多少篇文献？",
        verified_inventory_count=True,
    ) == answer


def test_topic_inventory_question_cannot_treat_candidate_count_as_library_count() -> None:
    out = finalize_runtime._normalize_retrieval_window_claims(
        "我的库里一共只有 2 篇文献讨论单像素成像。",
        prompt="我库里有几篇讨论单像素成像？",
    )

    assert "库里一共只有" not in out
    assert "本轮检索到 2 篇候选文献" in out


def test_english_library_candidate_counts_are_scoped_to_retrieval() -> None:
    exact = finalize_runtime._normalize_retrieval_window_claims(
        "There are exactly 2 papers in your library about SPI.",
        prompt="How many papers in my library discuss SPI?",
    )
    words = finalize_runtime._normalize_retrieval_window_claims(
        "Your library contains two papers about SPI.",
        prompt="How many papers in my library discuss SPI?",
    )

    assert "current retrieval found 2 candidate papers" in exact
    assert "current retrieval window contains two papers" in words


def test_negative_boundary_answer_clarifies_not_core_paper() -> None:
    answer = finalize_runtime._maybe_clarify_negative_boundary_answer(
        "**\u7ed3\u8bba\uff1a\u5173\u7cfb\u4e0d\u5927\uff0c\u4e0d\u5efa\u8bae\u4e00\u8d77\u8bfb\u3002** "
        "\u8fd9\u7bc7\u8bba\u6587\u662f\u7535\u9a71\u52a8\u9499\u949b\u77ff\u6fc0\u5149\u5668\u4ef6\u7814\u7a76\u3002",
        prompt=(
            "\u8fd9\u7bc7 perovskite laser \u548c\u6211\u7684\u5355\u50cf\u7d20\u6210\u50cf"
            "\u4e3b\u7ebf\u5173\u7cfb\u5927\u5417\uff1f\u503c\u5f97\u4e00\u8d77\u8bfb\u5417\uff1f"
        ),
    )

    assert "\u4e0d\u662f\u5f53\u524d\u4e3b\u7ebf\u7684\u6838\u5fc3\u6587\u732e" in answer
    assert "\u4e0d\u662f" in answer


def test_prompt_requested_reference_targets_accepts_naive_source_trace():
    labels = [
        label
        for label, _alts in finalize_runtime._prompt_requested_reference_targets(
            "ADMM 是怎么来的？作者这里是借鉴了谁的想法吗？ADMM-Net 又是谁先做的？"
        )
    ]

    assert labels == ["ADMM", "ADMM-Net"]


def test_prompt_requested_reference_targets_does_not_confuse_admm_net_with_admm():
    labels = [
        label
        for label, _alts in finalize_runtime._prompt_requested_reference_targets(
            "ADMM-Net \u4e4b\u524d\u662f\u8c01\u505a\u7684\uff1f\u6211\u60f3\u77e5\u9053\u8fd9\u6761\u7ebf\u7d22\u5e94\u8be5\u4ece\u54ea\u7bc7\u5de5\u4f5c\u770b\u8d77\u3002"
        )
    ]

    assert labels == ["ADMM-Net"]


def test_maybe_append_requested_refs_uses_admm_net_label_after_wrong_inline_ref(monkeypatch):
    monkeypatch.setattr(finalize_runtime, "_load_reference_index", lambda _path: {"loaded": True})

    def _resolve(_index, _source_path, ref_num, *, source_sha1=""):
        if int(ref_num) == 21:
            return {
                "ref": {
                    "title": "Deep tensor ADMM-Net for snapshot compressive imaging",
                    "authors": "Jiawei Ma",
                    "venue": "ICCV",
                    "year": "2019",
                }
            }
        return {}

    monkeypatch.setattr(finalize_runtime, "_resolve_reference_entry", _resolve)
    answer = "ADMM-Net modeled the decoding process as a tensor recovery problem [[CITE:s7f6b9404:31]]."
    out = finalize_runtime._maybe_append_prompt_requested_inpaper_refs(
        answer,
        prompt=(
            "ADMM-Net \u4e4b\u524d\u662f\u8c01\u505a\u7684\uff1f"
            "\u6211\u60f3\u77e5\u9053\u8fd9\u6761\u7ebf\u7d22\u5e94\u8be5\u4ece\u54ea\u7bc7\u5de5\u4f5c\u770b\u8d77\u3002"
        ),
        answer_hits=[{"meta": {"source_path": "paper.md"}}],
        db_dir=Path("db"),
        locked_citation_source=None,
    )

    assert "ADMM-Net [[CITE:" in out
    assert "\u539f\u8bba\u6587\u6765\u6e90\uff1aADMM [[CITE:" not in out


def test_finalize_generation_answer_runs_postprocess_validate_and_quality(monkeypatch):
    calls = []
    figure_kwargs = {}
    citation_plan = {
        "version": 1,
        "intent": "origin_lookup",
        "budget": {"system_a": 1, "system_b": 1},
        "slots": [{"preferred_system": "system_b", "candidate_refs": [35]}],
    }

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
        research_answer_plan="method_explain",
        paper_guide_contracts_seed={"citation_plan": citation_plan},
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
    assert out["answer_quality"]["citation_plan"] == citation_plan
    assert out["answer_quality"]["research_answer_plan"] == "method_explain"
    assert out["paper_guide_contracts"]["citation_plan"] == citation_plan
    assert out["paper_guide_contracts"]["intent"]["research_answer_plan"] == "method_explain"


def test_finalize_strips_model_system_b_marker_when_plan_disables_system_b(monkeypatch):
    citation_plan = {
        "version": 1,
        "intent": "method_explain",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [{"preferred_system": "system_a", "candidate_hits": [1]}],
    }
    monkeypatch.setattr(
        finalize_runtime,
        "detect_text_reference_opportunities",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("System B detection must not run when its budget is disabled")
        ),
    )

    out = finalize_runtime._finalize_generation_answer(
        "The method uses retrieved evidence [[CITE:s1234abcd:7]].",
        prompt="How does this method work?",
        prompt_for_user="How does this method work?",
        answer_hits=[{"text": "Retrieved method evidence.", "meta": {"source_path": "paper.md"}}],
        db_dir=Path("db"),
        locked_citation_source=None,
        answer_intent="reading",
        answer_depth="L2",
        answer_output_mode="reading_guide",
        paper_guide_mode=False,
        paper_guide_contract_enabled=False,
        paper_guide_prompt_family="method",
        paper_guide_special_focus_block="",
        paper_guide_focus_source_path="",
        paper_guide_direct_source_path="",
        paper_guide_bound_source_path="",
        paper_guide_candidate_refs_by_source={},
        paper_guide_support_slots=[],
        paper_guide_evidence_cards=[],
        paper_guide_contracts_seed={"citation_plan": citation_plan},
        apply_paper_guide_answer_postprocess=lambda answer, **_kwargs: (answer, []),
        maybe_append_library_figure_markdown=lambda answer, **_kwargs: answer,
        validate_structured_citations=lambda answer, **_kwargs: (answer, {"kept": 1}),
    )

    assert "[[CITE:" not in out["answer"]
    assert "retrieved evidence" in out["answer"]


def test_finalize_keeps_precomputed_origin_reference_candidates(monkeypatch):
    citation_plan = {
        "version": 1,
        "intent": "origin_lookup",
        "budget": {"system_a": 1, "system_b": 1},
        "slots": [{"preferred_system": "system_b", "candidate_refs": [50]}],
    }
    opportunity = {
        "sid": "s1234abcd",
        "ref_num": 50,
        "source_path": "scinerf.md",
        "label": "snapshot compressive imaging",
        "evidence_quote": "video Snapshot Compressive Imaging (SCI) [50] system has emerged",
    }
    monkeypatch.setattr(
        finalize_runtime,
        "detect_text_reference_opportunities",
        lambda **_kwargs: pytest.fail("precomputed origin reference must not be replaced after generation"),
    )
    seen: dict[str, object] = {}

    def _validate(answer, **kwargs):
        seen.update(kwargs)
        return answer, {"kept": 1}

    finalize_runtime._finalize_generation_answer(
        "The SCI lineage builds on earlier snapshot compression [[CITE:s1234abcd:38]].",
        prompt="SCI 是怎么从光谱成像走到 3D 场景重建的？",
        prompt_for_user="SCI 是怎么从光谱成像走到 3D 场景重建的？",
        answer_hits=[{"text": opportunity["evidence_quote"], "meta": {"source_path": "scinerf.md"}}],
        db_dir=Path("db"),
        locked_citation_source=None,
        answer_intent="reading",
        answer_depth="L2",
        answer_output_mode="reading_guide",
        paper_guide_mode=False,
        paper_guide_contract_enabled=False,
        paper_guide_prompt_family="overview",
        paper_guide_special_focus_block="",
        paper_guide_focus_source_path="",
        paper_guide_direct_source_path="",
        paper_guide_bound_source_path="",
        paper_guide_candidate_refs_by_source={"scinerf.md": [50]},
        paper_guide_support_slots=[],
        paper_guide_evidence_cards=[],
        paper_guide_contracts_seed={
            "citation_plan": citation_plan,
            "reference_opportunities": [opportunity],
        },
        apply_paper_guide_answer_postprocess=lambda answer, **_kwargs: (answer, []),
        maybe_append_library_figure_markdown=lambda answer, **_kwargs: answer,
        validate_structured_citations=_validate,
    )

    assert seen["paper_guide_candidate_refs_by_source"] == {"scinerf.md": [50]}


def test_finalize_fast_exact_reuses_support_without_full_text_rescan(monkeypatch):
    support = {
        "source_path": "paper.md",
        "heading_path": "2. Related Work",
        "block_id": "blk_admm",
        "anchor_id": "p_admm",
        "locate_anchor": "Most existing methods employ ADMM [4].",
        "ref_nums": [4],
        "resolved_ref_num": 4,
    }
    monkeypatch.setattr(
        finalize_runtime,
        "detect_text_reference_opportunities",
        lambda **_kwargs: pytest.fail("fast exact path must not rescan source text"),
    )
    monkeypatch.setattr(
        finalize_runtime,
        "_build_answer_quality_probe",
        lambda answer, **_kwargs: {"minimum_ok": True, "answer": answer},
    )

    out = finalize_runtime._finalize_generation_answer(
        "ADMM is established prior work [[CITE:s1234abcd:4]].\n> Most existing methods employ ADMM [4].",
        prompt="Which reference is cited for ADMM, and where exactly?",
        prompt_for_user="Which reference is cited for ADMM, and where exactly?",
        answer_hits=[{"text": support["locate_anchor"], "meta": {"source_path": "paper.md"}}],
        db_dir=Path("db"),
        locked_citation_source={"sid": "s1234abcd", "source_path": "paper.md"},
        answer_intent="reading",
        answer_depth="L2",
        answer_output_mode="reading_guide",
        paper_guide_mode=True,
        paper_guide_contract_enabled=False,
        paper_guide_prompt_family="citation_lookup",
        paper_guide_special_focus_block="",
        paper_guide_focus_source_path="paper.md",
        paper_guide_direct_source_path="paper.md",
        paper_guide_bound_source_path="paper.md",
        paper_guide_candidate_refs_by_source={"paper.md": [4]},
        paper_guide_support_slots=[support],
        paper_guide_evidence_cards=[],
        paper_guide_precomputed_support_resolution=[support],
        paper_guide_fast_exact=True,
        apply_paper_guide_answer_postprocess=lambda *_args, **_kwargs: pytest.fail(
            "fast exact path must reuse precomputed support"
        ),
        maybe_append_library_figure_markdown=lambda answer, **_kwargs: answer,
        validate_structured_citations=lambda answer, **_kwargs: (answer, {"kept": 1}),
    )

    assert "[[CITE:s1234abcd:4]]" in out["answer"]
    assert "> Most existing methods employ ADMM." in out["answer"]
    assert "> Most existing methods employ ADMM [4]." not in out["answer"]
    assert out["paper_guide_support_resolution"][0]["block_id"] == "blk_admm"
    system_a = out["paper_guide_contracts"]["render_packet"]["cite_details"][0]
    assert system_a["citation_route"] == "system_a"
    assert system_a["block_id"] == "blk_admm"
    assert system_a["anchor_id"] == "p_admm"


def test_finalize_fast_exact_honors_disabled_system_b_budget(monkeypatch):
    support = {
        "source_path": "paper.md",
        "heading_path": "3. Degradation Model",
        "block_id": "blk_model",
        "anchor_id": "p_model",
        "locate_anchor": "The observation includes blur and additive noise [4].",
        "resolved_ref_num": 4,
    }
    seen: dict[str, object] = {}

    def _validate(answer, **kwargs):
        seen.update(kwargs)
        return answer, {"kept": 0}

    out = finalize_runtime._finalize_generation_answer(
        "The model includes blur and noise [[CITE:s1234abcd:4]].",
        prompt="Explain the degradation model with supporting citations.",
        prompt_for_user="Explain the degradation model with supporting citations.",
        answer_hits=[{"text": support["locate_anchor"], "meta": {"source_path": "paper.md"}}],
        db_dir=Path("db"),
        locked_citation_source={"sid": "s1234abcd", "source_path": "paper.md"},
        answer_intent="reading",
        answer_depth="L2",
        answer_output_mode="reading_guide",
        paper_guide_mode=True,
        paper_guide_contract_enabled=False,
        paper_guide_prompt_family="method",
        paper_guide_special_focus_block="",
        paper_guide_focus_source_path="paper.md",
        paper_guide_direct_source_path="paper.md",
        paper_guide_bound_source_path="paper.md",
        paper_guide_candidate_refs_by_source={"paper.md": [4]},
        paper_guide_support_slots=[support],
        paper_guide_evidence_cards=[],
        paper_guide_precomputed_support_resolution=[support],
        paper_guide_fast_exact=True,
        paper_guide_contracts_seed={
            "citation_plan": {
                "intent": "evidence_lookup",
                "budget": {"system_a": 1, "system_b": 0},
            }
        },
        apply_paper_guide_answer_postprocess=lambda *_args, **_kwargs: pytest.fail(
            "fast exact path must reuse precomputed support"
        ),
        maybe_append_library_figure_markdown=lambda answer, **_kwargs: answer,
        validate_structured_citations=_validate,
    )

    assert "[[CITE:" not in out["answer"]
    assert seen["paper_guide_candidate_refs_by_source"] == {}
    assert out["answer_quality"]["citation_plan"]["budget"]["system_b"] == 0


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


def test_contract_snapshot_drops_stale_seed_render_text_when_final_answer_changes():
    contracts = finalize_runtime._build_paper_guide_contract_snapshot(
        paper_guide_mode=False,
        intent_model=None,
        answer_markdown="Final four-step route.",
        final_answer_markdown="Final four-step route.",
        evidence_cards=[],
        candidate_refs_by_source={},
        support_slots=[],
        support_resolution=[],
        needs_supplement=False,
        citation_validation={},
        doc_list_contract=[{"source_path": "db/paper-a.md", "source_name": "Paper A"}],
        paper_guide_contracts_seed={
            "render_packet": {
                "answer_markdown": "Simplified document list.",
                "rendered_body": "Stale simplified document list.",
                "rendered_content": "Stale rendered content.",
                "copy_markdown": "Stale copy markdown.",
                "copy_text": "Stale copy text.",
            }
        },
    )

    packet = contracts["render_packet"]
    assert packet["answer_markdown"] == "Final four-step route."
    assert packet["rendered_body"] == ""
    assert packet["rendered_content"] == ""
    assert packet["copy_markdown"] == ""
    assert packet["copy_text"] == ""


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

    assert "[[CITE:s1234abcd:26]]" in out["answer"]


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
    assert "[2]" in out["answer"]
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


def test_finalize_generation_answer_sanitizes_inline_internal_doc_labels(monkeypatch):
    monkeypatch.setattr(finalize_runtime, "_reconcile_kb_notice", lambda answer, **kwargs: answer)
    monkeypatch.setattr(finalize_runtime, "_apply_answer_contract_v1", lambda answer, **kwargs: answer)
    monkeypatch.setattr(finalize_runtime, "_enhance_kb_miss_fallback", lambda answer, **kwargs: answer)
    monkeypatch.setattr(finalize_runtime, "_build_answer_quality_probe", lambda answer, **kwargs: {"minimum_ok": True, "answer": answer})

    raw = (
        "Among the retrieved papers, the following two mention NeRF:\n\n"
        "**DOC-1** (): *CVPR-2024-SCINeRF: Neural Radiance Fields from a Snapshot Compressive Image*\n\n"
        "The paper repeatedly uses NeRF as its underlying scene representation.\n\n"
        "**DOC-2** (): *ICIP-2025-SCIGS: 3D Gaussians Splatting from A Snapshot Compressive Image*\n\n"
        "It explicitly contrasts the limitations of NeRF-based reconstruction methods.\n\n"
        "The remaining papers (DOC-3, DOC-4) do not mention NeRF."
    )

    out = finalize_runtime._finalize_generation_answer(
        raw,
        prompt="Which papers in my library mention NeRF?",
        prompt_for_user="Which papers in my library mention NeRF?",
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

    assert "DOC-1" not in out["answer"]
    assert "DOC-2" not in out["answer"]
    assert "DOC-3" not in out["answer"]
    assert "DOC-4" not in out["answer"]
    assert "CVPR-2024-SCINeRF" in out["answer"]
    assert "ICIP-2025-SCIGS" in out["answer"]
    assert "The remaining papers do not mention NeRF." in out["answer"]


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


def test_finalize_generation_answer_preserves_rich_reading_route_with_internal_doc_labels(monkeypatch):
    monkeypatch.setattr(finalize_runtime, "_reconcile_kb_notice", lambda answer, **kwargs: answer)
    monkeypatch.setattr(finalize_runtime, "_apply_answer_contract_v1", lambda answer, **kwargs: answer)
    monkeypatch.setattr(finalize_runtime, "_enhance_kb_miss_fallback", lambda answer, **kwargs: answer)
    monkeypatch.setattr(
        finalize_runtime,
        "_build_answer_quality_probe",
        lambda answer, **kwargs: {"minimum_ok": True, "answer": answer},
    )

    raw = """# 单像素成像入门主线：先读这3篇

## 1. 第一篇：综述 — 建立全局认知

**推荐论文（DOC-1）：** *Advances and Challenges of Single-Pixel Imaging Based on Deep Learning*

**主要看什么：** 先看 Fundamentals，理解调制、测量和重建的基本框架 [1]。

**为什么先读：** 它先给出问题、方法、进展和挑战的全局地图。

## 2. 第二篇：原理对比 — 理解确定性方法

**推荐论文（DOC-2）：** *Hadamard single-pixel imaging versus Fourier single-pixel imaging*

**主要看什么：** 对比 HSI 与 FSI 的原理、成像效率和噪声鲁棒性 [2]。

**为什么接着读：** 它把综述中的抽象分类落到硬件和采样策略选择上。

## 3. 第三篇：系统展望 — 理解技术边界

**推荐论文（DOC-3）：** *Principles and prospects for single-pixel imaging*

**主要看什么：** 看采集与重建策略、扫描效率以及适用波段 [3]。

**为什么收尾：** 它帮助判断什么场景下 SPI 真正有优势。

## 阅读顺序建议

综述（全局地图）→ 原理对比（核心方法）→ 系统展望（边界判断）。
"""
    docs = [
        (
            r"db\LPR\LPR.en.md",
            "Advances and Challenges of Single-Pixel Imaging Based on Deep Learning.pdf",
            "Fundamentals of Single-Pixel Imaging",
        ),
        (
            r"db\OE\OE.en.md",
            "Hadamard single-pixel imaging versus Fourier single-pixel imaging.pdf",
            "2.1 Principle of HSI and FSI",
        ),
        (
            r"db\NatPhoton\NatPhoton.en.md",
            "Principles and prospects for single-pixel imaging.pdf",
            "Acquisition and image reconstruction strategies",
        ),
    ]
    answer_hits = [
        {
            "text": f"Grounded evidence for {source_name}",
            "meta": {"source_path": source_path, "ref_best_heading_path": heading},
        }
        for source_path, source_name, heading in docs
    ]
    evidence_cards = [
        {
            "source_path": source_path,
            "primary_evidence": {
                "source_path": source_path,
                "source_name": source_name,
                "heading_path": heading,
                "snippet": f"Grounded evidence for {source_name}",
            },
        }
        for source_path, source_name, heading in docs
    ]

    out = finalize_runtime._finalize_generation_answer(
        raw,
        prompt="我刚开始看单像素成像，想先建立主线，只推荐3篇并给出阅读顺序和引用。",
        prompt_for_user="我刚开始看单像素成像，想先建立主线，只推荐3篇并给出阅读顺序和引用。",
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

    answer = str(out.get("answer") or "")
    assert "DOC-" not in answer
    assert "为什么先读" in answer
    assert "主要看什么" in answer
    assert "阅读顺序建议" in answer
    assert "[1]" in answer
    assert "[2]" in answer
    assert "[3]" in answer
    assert "根据命中的库内文献" not in answer
    assert [
        item.get("source_path")
        for item in list(dict(out.get("paper_guide_contracts") or {}).get("doc_list") or [])
    ] == [source_path for source_path, _source_name, _heading in docs]


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


def test_build_multi_paper_doc_list_contract_keeps_complete_primary_evidence_beyond_short_summary():
    source_path = r"db\OE-2017-HSI-FSI\OE-2017-HSI-FSI.en.md"
    raw_snippet = (
        "## 2. Comparison of theory\n"
        "Hadamard single-pixel imaging and Fourier single-pixel imaging are representative deterministic methods. "
        "The paper compares their principles, imaging efficiency, and noise robustness under the same experimental setup. "
        "Hadamard basis patterns are binary, which makes them suitable for high-speed modulation by a digital micromirror device."
    )

    out = finalize_runtime._build_multi_paper_doc_list_contract(
        prompt="Which papers should I read to compare HSI and FSI?",
        seed_docs=[
            {
                "text": raw_snippet,
                "meta": {
                    "source_path": source_path,
                    "ref_best_heading_path": "2. Comparison of theory",
                    "ref_show_snippets": [raw_snippet],
                },
            }
        ],
        answer_hits=[],
        evidence_cards=[],
        apply_prompt_filter=False,
    )

    row = out[0]
    primary = dict(row.get("primary_evidence") or {})
    assert len(str(row.get("summary_line") or "")) <= 180
    assert len(str(primary.get("snippet") or "")) > len(str(row.get("summary_line") or ""))
    assert str(primary.get("snippet") or "").endswith("device.")


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


def test_build_multi_paper_doc_list_contract_carries_llm_pack_copy_from_ref_pack():
    source_path = (
        r"db\CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image"
        r"\CVPR-2024-SCINeRF- Neural Radiance Fields from a Snapshot Compressive Image.en.md"
    )

    out = finalize_runtime._build_multi_paper_doc_list_contract(
        prompt="Which papers in my library mention SCI (Snapshot Compressive Imaging)?",
        seed_docs=[
            {
                "text": "SCI-based 3D scene reconstruction recovers a scene from a single compressed capture.",
                "meta": {
                    "source_path": source_path,
                    "ref_best_heading_path": "Abstract",
                    "ref_show_snippets": [
                        "SCI-based 3D scene reconstruction recovers a scene from a single compressed capture."
                    ],
                    "ref_pack": {
                        "what": "The paper studies SCI-based 3D scene reconstruction from a single compressed capture rather than only introducing the term.",
                        "why": "It explicitly frames the method as Snapshot Compressive Imaging (SCI), so it is a direct match for papers that mention SCI.",
                    },
                },
            }
        ],
        answer_hits=[],
        evidence_cards=[],
    )

    assert len(out) == 1
    row = out[0]
    assert row["summary_line"].startswith("The paper studies SCI-based 3D scene reconstruction")
    assert row["summary_generation"] == "llm_pack"
    assert row["why_line"].startswith("It explicitly frames the method as Snapshot Compressive Imaging")
    assert row["why_generation"] == "llm_pack"


def test_filter_multi_paper_doc_list_contract_respects_requested_count():
    rows = [
        {
            "source_path": f"db/paper-{idx}.md",
            "source_name": f"Paper {idx}.pdf",
            "heading_path": "Abstract",
            "summary_line": f"Snapshot Compressive Imaging (SCI) contribution {idx} with reconstruction evidence.",
        }
        for idx in range(1, 7)
    ]

    out = finalize_runtime._filter_multi_paper_doc_list_contract(
        prompt="List 4 papers that mention SCI.",
        doc_list=rows,
    )

    assert len(out) == 4


def test_multi_paper_answer_preserves_complete_requested_route() -> None:
    answer = "\n".join(
        [
            "下面按阅读顺序给出 4 篇论文：",
            "1. **Paper A** - 综述与基本原理。",
            "2. **Paper B** - 采样与重建。",
            "3. **Paper C** - 深度学习重建。",
            "4. **Paper D** - 实时系统实现。",
            "每一步都对应可核对的库内依据。",
        ]
    )

    assert finalize_runtime._multi_paper_answer_needs_contract_rebuild(
        answer=answer,
        prompt="请只用最相关的 4 篇论文做阅读路线。",
    ) is False


def test_multi_paper_answer_rebuilds_wrong_requested_count() -> None:
    answer = "\n".join(
        [
            "下面给出论文路线：",
            "1. **Paper A** - overview and evidence.",
            "2. **Paper B** - sampling and reconstruction.",
            "3. **Paper C** - learned reconstruction.",
            "4. **Paper D** - real-time system.",
            "5. **Paper E** - unrelated extra item.",
            "6. **Paper F** - another unrelated extra item.",
        ]
    )

    assert finalize_runtime._multi_paper_answer_needs_contract_rebuild(
        answer=answer,
        prompt="Please use only 4 papers.",
    ) is True


def test_multi_paper_count_accepts_numbered_markdown_headings() -> None:
    answer = "\n".join(
        f"## {idx}. Step {idx}\nEvidence-backed reading rationale for this paper [{idx}]"
        for idx in range(1, 5)
    )

    assert finalize_runtime._count_multi_paper_answer_items(answer) == 4
    assert finalize_runtime._multi_paper_answer_needs_contract_rebuild(
        answer=answer,
        prompt="Use only 4 papers and cite each source.",
    ) is False


def test_multi_paper_count_accepts_chinese_step_headings() -> None:
    answer = "\n".join(
        f"## \u7b2c{idx}\u6b65\uff1aRoute stage {idx}\nEvidence-backed rationale [{idx}]"
        for idx in range(1, 5)
    )

    assert finalize_runtime._count_multi_paper_answer_items(answer) == 4
    assert finalize_runtime._multi_paper_answer_needs_contract_rebuild(
        answer=answer,
        prompt="Use only 4 papers and cite each source.",
    ) is False


def test_multi_paper_count_accepts_chinese_paper_headings_without_rebuild() -> None:
    answer = """# \u5355\u50cf\u7d20\u6210\u50cf\u5165\u95e8\uff1a\u4e09\u7bc7\u63a8\u8350\u9605\u8bfb\u987a\u5e8f

## \u7b2c1\u7bc7\uff1aPrinciples and prospects for single-pixel imaging
\u4ece\u57fa\u672c\u539f\u7406\u5efa\u7acb\u5b8c\u6574\u77e5\u8bc6\u6846\u67b6\u3002

## \u7b2c2\u7bc7\uff1a3D single-pixel video
\u7406\u89e3\u5b9e\u9645\u7cfb\u7edf\u7684\u786c\u4ef6\u7ea6\u675f\u548c\u6743\u8861\u3002

## \u7b2c3\u7bc7\uff1aAdvances and Challenges of Single-Pixel Imaging Based on Deep Learning
\u4e86\u89e3\u6781\u4f4e\u91c7\u6837\u7387\u4e0b\u7684\u5b66\u4e60\u578b\u91cd\u5efa\u524d\u6cbf\u3002
"""

    assert finalize_runtime._count_multi_paper_answer_items(answer) == 3
    assert finalize_runtime._multi_paper_answer_needs_contract_rebuild(
        answer=answer,
        prompt="\u8bf7\u4ece\u5e93\u91cc\u90093\u7bc7\u6700\u9002\u5408\u6309\u987a\u5e8f\u9605\u8bfb\u7684\u8bba\u6587\u3002",
    ) is False


def test_requested_multi_paper_repair_removes_extra_recommendation_and_restores_source_marker() -> None:
    answer = """## 1. Overview
Paper: Principles and prospects for single-pixel imaging
Evidence without a marker.

---

## 2. Sampling
Paper: Sequentially designed compressed sensing [2]

---

## 3. Deep learning
Paper: Part-based image-loop network [3]

---

## 4. System
Paper: 3D single-pixel video [4]

---

**Further reading:** A fifth paper [5].
"""
    hits = [
        {
            "text": "overview evidence",
            "meta": {"source_path": "db/NatPhoton-2019-Principles and prospects for single-pixel imaging.en.md"},
        },
        {"text": "sampling evidence", "meta": {"source_path": "db/Sequentially designed compressed sensing.md"}},
        {"text": "learning evidence", "meta": {"source_path": "db/Part-based image-loop network.md"}},
        {"text": "system evidence", "meta": {"source_path": "db/3D single-pixel video.md"}},
    ]

    repaired = finalize_runtime._repair_requested_multi_paper_answer(
        answer,
        prompt="Use only 4 papers and cite each source so I can verify the evidence.",
        answer_hits=hits,
    )

    assert "Further reading" not in repaired
    assert "Evidence without a marker. [1]" in repaired
    assert finalize_runtime._multi_paper_answer_needs_contract_rebuild(
        answer=repaired,
        prompt="Use only 4 papers and cite each source so I can verify the evidence.",
    ) is False

    heading_extra = answer.replace("**Further reading:**", "## Further reading")
    assert "Further reading" not in finalize_runtime._strip_requested_multi_paper_extras(heading_extra)


def test_multi_paper_repair_without_explicit_count_drops_followup_papers_but_keeps_reading_advice() -> None:
    answer = """对于刚入门 SPI，建议按以下顺序阅读三篇核心文献：

### 1. 先读综述
主要看原理和系统边界 [3]。

### 2. 再读方法对比
主要看 Hadamard 与 Fourier 的差异 [2]。

### 3. 最后看前沿
主要看深度学习重建与泛化挑战 [1]。

### 阅读建议
- **顺序：** 综述 → 方法对比 → 深度学习前沿。
- **重点：** 每篇都看动机、方法和局限。
- **后续：** 读完后还可以选择其他方向：
 - 自适应采样可读额外论文 [5]。
 - 3D 成像可读额外论文 [6]。
 - 自监督网络可读额外论文 [4]。
"""

    out = finalize_runtime._repair_requested_multi_paper_answer(
        answer,
        prompt="我刚开始看单像素成像，想先建立主线，应该先读哪几篇？每篇主要看什么？",
        answer_hits=[],
    )

    assert "先读综述" in out
    assert "再读方法对比" in out
    assert "最后看前沿" in out
    assert "### 阅读建议" in out
    assert "**顺序：**" in out
    assert "**重点：**" in out
    assert "**后续：**" not in out
    assert "[4]" not in out
    assert "[5]" not in out
    assert "[6]" not in out


def test_multi_paper_contract_drops_limitations_with_unselected_paper_recommendations() -> None:
    answer = """## 1. Review
Read the overview [3].

## 2. Comparison
Compare the two acquisition strategies [2].

## 3. Frontier
Study the learning-based reconstruction limits [1].

## Limitations
For 3D imaging, read another paper [5]. A self-supervised route is covered by [4].

## Next action
Read [3], then [2], and finish with [1]."""

    out = finalize_runtime._strip_multi_paper_unselected_recommendation_sections(
        answer,
        allowed_citation_nums={1, 2, 3},
    )

    assert "## Limitations" not in out
    assert "[4]" not in out
    assert "[5]" not in out
    assert "## Next action" in out
    assert "Read [3], then [2], and finish with [1]." in out


def test_multi_paper_contract_selects_chinese_ordinal_core_sections_and_drops_followup_clause() -> None:
    source_paths = ["learning.md", "comparison.md", "prospects.md", "foveated.md"]
    hits = [
        {"text": "evidence", "meta": {"source_path": source_path}}
        for source_path in source_paths
    ]
    docs = [
        {"source_path": source_path, "source_name": source_path}
        for source_path in source_paths
    ]
    answer = """## 第一篇：建立整体框架
**prospects** [3]

## 第二篇：理解主流方法
**comparison** [2]

## 第三篇：了解学习方法
**learning** [1]

## 阅读路线图总结
读完这三篇就能建立主线。之后可根据兴趣深入自适应采样 [4]。"""

    selected = finalize_runtime._select_multi_paper_doc_list_from_answer(
        answer=answer,
        answer_hits=hits,
        doc_list=docs,
    )
    assert [row["citation_num"] for row in selected] == [3, 2, 1]
    assert [row["source_path"] for row in selected] == ["prospects.md", "comparison.md", "learning.md"]

    out = finalize_runtime._strip_multi_paper_unselected_recommendation_sections(
        answer,
        allowed_citation_nums={1, 2, 3},
    )
    assert "读完这三篇就能建立主线。" in out
    assert "自适应采样" not in out
    assert "[4]" not in out


def test_multi_paper_contract_selects_bold_chinese_ordinal_sections_with_descriptors() -> None:
    source_paths = ["learning.md", "comparison.md", "prospects.md"]
    hits = [
        {"text": "evidence", "meta": {"source_path": source_path}}
        for source_path in source_paths
    ]
    answer = """**第一篇必读：** **prospects** [3]

Read the field overview.

**第二篇（方法对比）：** **comparison** [2]

Compare deterministic coding methods.

**第三篇（综述）：** **learning** [1]

Study the learning frontier.

```
第一步：prospects
第二步：comparison
第三步：learning
```"""

    selected = finalize_runtime._select_multi_paper_doc_list_from_answer(
        answer=answer,
        answer_hits=hits,
        doc_list=[{"source_path": path} for path in source_paths],
    )

    assert finalize_runtime._count_multi_paper_answer_items(answer) == 3
    assert [row["citation_num"] for row in selected] == [3, 2, 1]


def test_multi_paper_contract_drops_bold_advanced_tip_with_unselected_sources() -> None:
    answer = """## 1. Overview
Read the overview [3].

## Reading plan
Read the three core papers in order.

**Advanced tip:** Continue with self-supervised reconstruction [4].
- Then study 3D imaging [5]."""

    out = finalize_runtime._strip_multi_paper_unselected_recommendation_sections(
        answer,
        allowed_citation_nums={1, 2, 3},
    )

    assert "## Reading plan" in out
    assert "Advanced tip" not in out
    assert "[4]" not in out
    assert "[5]" not in out


def test_multi_paper_contract_drops_embedded_bullet_with_unselected_source() -> None:
    answer = """## 3. Learning frontier
- Deep learning improves reconstruction quality [1].
- A self-supervised image-loop network is another paper [4].

## Summary
The three core papers form the requested roadmap.

**Advanced tip:** Study 3D imaging [6]."""

    out = finalize_runtime._strip_multi_paper_unselected_recommendation_sections(
        answer,
        allowed_citation_nums={1, 2, 3},
    )

    assert "improves reconstruction quality [1]" in out
    assert "self-supervised" not in out
    assert "Advanced tip" not in out
    assert "[4]" not in out
    assert "[6]" not in out


def test_single_paper_selection_strips_other_candidate_table_but_keeps_reading_locations() -> None:
    answer = (
        "# 最直接的比较论文\n\n"
        "**Hadamard single-pixel imaging versus Fourier single-pixel imaging**\n\n"
        "## 为什么选这篇\n\n标题和实验都直接比较 HSI 与 FSI。\n\n"
        "---\n\n## 其他候选论文为何不选\n\n"
        "| 论文 | 不选原因 |\n|---|---|\n| 综述 | 不够直接 |\n\n"
        "---\n\n## 关键阅读位置\n\n- 第2节：理论对比"
    )

    out = finalize_runtime._strip_single_paper_selection_extras(answer)

    assert "其他候选论文" not in out
    assert "综述" not in out
    assert "关键阅读位置" in out
    assert "第2节" in out


def test_single_library_paper_selection_does_not_detect_system_b_opportunities(monkeypatch) -> None:
    def _unexpected_detection(**_kwargs):
        raise AssertionError("System B opportunity detection must not run for a library-paper pick")

    monkeypatch.setattr(
        finalize_runtime,
        "detect_text_reference_opportunities",
        _unexpected_detection,
    )
    out = finalize_runtime._finalize_generation_answer(
        (
            "# Best paper\n\nThe direct match is the OE 2017 comparison paper [[CITE:s1234abcd:26]].\n\n"
            "## Other candidates\n\n| Paper | Why not |\n|---|---|\n| Review | Less direct |\n\n"
            "## Reading location\n\nSection 2 compares both methods."
        ),
        prompt="Which paper in my library directly compares HSI and FSI? Only give 1 paper.",
        prompt_for_user="Which paper in my library directly compares HSI and FSI? Only give 1 paper.",
        answer_hits=[{"text": "HSI and FSI are compared directly.", "meta": {"source_path": "oe2017.en.md"}}],
        db_dir=Path("db"),
        locked_citation_source=None,
        answer_intent="reading",
        answer_depth="L2",
        answer_output_mode="reading_guide",
        paper_guide_mode=False,
        paper_guide_contract_enabled=False,
        paper_guide_prompt_family="compare",
        paper_guide_special_focus_block="",
        paper_guide_focus_source_path="",
        paper_guide_direct_source_path="",
        paper_guide_bound_source_path="",
        paper_guide_candidate_refs_by_source={},
        paper_guide_support_slots=[],
        paper_guide_evidence_cards=[],
        paper_guide_contracts_seed={},
        paper_guide_retrieval_confidence_hint={},
        apply_paper_guide_answer_postprocess=lambda answer, **_kwargs: (answer, []),
        maybe_append_library_figure_markdown=lambda answer, **_kwargs: answer,
        validate_structured_citations=lambda answer, **_kwargs: (answer, {"kept": 0}),
    )

    answer = str(out.get("answer") or "")
    assert "Other candidates" not in answer
    assert "Reading location" in answer
    assert "[[CITE:" not in answer
    assert out["answer_quality"]["requested_paper_count"] == 1
    assert out["answer_quality"]["actual_paper_count"] == 1
    assert out["answer_quality"]["paper_count_ok"] is True


def test_multi_paper_llm_summary_with_foreign_technical_marker_falls_back_to_evidence() -> None:
    out = finalize_runtime._build_multi_paper_doc_list_contract(
        prompt="Which papers discuss single-pixel imaging?",
        seed_docs=[
            {
                "text": "Single-pixel imaging uses compressive sensing for acquisition and image reconstruction strategies.",
                "meta": {
                    "source_path": "db/natphoton-review.md",
                    "ref_best_heading_path": "Abstract",
                    "ref_show_snippets": [
                        "Single-pixel imaging uses compressive sensing for acquisition and image reconstruction strategies."
                    ],
                    "ref_pack": {
                        "what": "The paper implements an API using a dynamic-link library.",
                        "why": "It is useful background.",
                    },
                },
            }
        ],
        answer_hits=[],
        evidence_cards=[],
    )

    assert len(out) == 1
    assert "compressive sensing" in out[0]["summary_line"].lower()
    assert "API" not in out[0]["summary_line"]


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


def test_filter_multi_paper_doc_list_contract_keeps_cjk_adjacent_nerf_mentions():
    prompt = "Which papers in my library mention NeRF?"
    rows = [
        {
            "source_path": r"db\ICIP-2025-SCIGS\ICIP-2025-SCIGS.en.md",
            "source_name": "ICIP-2025-SCIGS- 3D Gaussians Splatting from A Snapshot Compressive Image.pdf",
            "heading_path": "Abstract",
            "summary_line": (
                "\u8be5\u8bba\u6587\u5728\u6458\u8981\u4e2d\u63d0\u53caNeRF-based reconstruction methods\uff0c"
                "\u5e76\u6307\u51fa\u5176\u5728\u52a8\u6001\u573a\u666f\u4e2d\u4ecd\u6709\u5c40\u9650\u3002"
            ),
            "primary_evidence": {
                "heading_path": "Abstract",
                "snippet": (
                    "Snapshot Compressive Imaging (SCI) offers a possibility for capturing information in "
                    "high-speed dynamic scenes. Despite promising results, current deep learning-based and "
                    "NeRF-based reconstruction methods still face limitations in handling dynamic scenes."
                ),
            },
        },
        {
            "source_path": r"db\CVPR-2024-SCINeRF\CVPR-2024-SCINeRF.en.md",
            "source_name": "CVPR-2024-SCINeRF- Neural Radiance Fields from A Snapshot Compressive Image.pdf",
            "heading_path": "Abstract",
            "summary_line": "SCINeRF exploits neural radiance fields as its underlying scene representation.",
            "primary_evidence": {
                "heading_path": "Abstract",
                "snippet": "Our approach builds upon the powerful 3D scene representation capabilities of neural radiance fields (NeRF).",
            },
        },
        {
            "source_path": r"db\Unrelated\Unrelated.en.md",
            "source_name": "Unrelated-3D Gaussian Splatting.pdf",
            "heading_path": "Abstract",
            "summary_line": "This paper discusses dynamic scene reconstruction with 3D Gaussian splatting.",
        },
    ]

    out = finalize_runtime._filter_multi_paper_doc_list_contract(
        prompt=prompt,
        doc_list=rows,
    )

    assert [item["source_name"] for item in out] == [
        "ICIP-2025-SCIGS- 3D Gaussians Splatting from A Snapshot Compressive Image.pdf",
        "CVPR-2024-SCINeRF- Neural Radiance Fields from A Snapshot Compressive Image.pdf",
    ]


def test_filter_multi_paper_doc_list_contract_excludes_cjk_adjacent_negated_nerf_mentions():
    prompt = "Which papers in my library mention NeRF?"
    rows = [
        {
            "source_path": r"db\Negative\Negative.en.md",
            "source_name": "Negative-3DGS-note.pdf",
            "heading_path": "Abstract",
            "summary_line": "\u672c\u6587\u672a\u63d0\u53caNeRF\uff0c\u4ec5\u8ba8\u8bba3D Gaussian Splatting\u3002",
        },
        {
            "source_path": r"db\CVPR-2024-SCINeRF\CVPR-2024-SCINeRF.en.md",
            "source_name": "CVPR-2024-SCINeRF- Neural Radiance Fields from A Snapshot Compressive Image.pdf",
            "heading_path": "Abstract",
            "summary_line": "SCINeRF exploits neural radiance fields as its underlying scene representation.",
            "primary_evidence": {
                "heading_path": "Abstract",
                "snippet": "Our approach builds upon the powerful 3D scene representation capabilities of neural radiance fields (NeRF).",
            },
        },
    ]

    out = finalize_runtime._filter_multi_paper_doc_list_contract(
        prompt=prompt,
        doc_list=rows,
    )

    assert [item["source_name"] for item in out] == [
        "CVPR-2024-SCINeRF- Neural Radiance Fields from A Snapshot Compressive Image.pdf",
    ]


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


def test_select_multi_paper_doc_list_follows_answer_titles_and_non_contiguous_canonical_markers():
    names = [
        "Frequency-division-multiplexed single-pixel imaging with metamaterials",
        "Advances and Challenges of Single-Pixel Imaging Based on Deep Learning",
        "Part-based image-loop network for single-pixel imaging",
        "Imaging biological tissue with single-pixel compressive holography",
        "Robust real-time single-pixel imaging based on a spinning mask via differential detection supplement",
        "Hadamard single-pixel imaging versus Fourier single-pixel imaging",
    ]
    hits = [
        {"text": name, "meta": {"source_path": f"db/{name}.en.md"}}
        for name in names
    ]
    doc_list = [
        {
            "source_path": f"db/{name}.en.md",
            "source_name": name,
            "summary_line": f"Evidence for {name}",
        }
        for name in names
    ]
    answer = """## 1. 综述与基本原理
**论文：** Advances and Challenges of Single-Pixel Imaging Based on Deep Learning [2]

## 2. 采样与重建
**论文：** Hadamard single-pixel imaging versus Fourier single-pixel imaging [6]

## 3. 深度学习重建
**论文：** Part-based image-loop network for single-pixel imaging [3]

## 4. 实时系统
**论文：** Robust real-time single-pixel imaging based on a spinning mask via differential detection [5]
"""

    selected = finalize_runtime._select_multi_paper_doc_list_from_answer(
        answer=answer,
        answer_hits=hits,
        doc_list=doc_list,
    )

    assert [item["citation_num"] for item in selected] == [2, 6, 3, 5]
    assert [item["source_name"] for item in selected] == [names[1], names[5], names[2], names[4]]


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

    assert "[[CITE:s1234abcd:35]]" in out["answer"]
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
