from pathlib import Path

import pytest

import kb.generation_answer_finalize_runtime as finalize_runtime


def test_collapse_single_item_numbered_block_after_evidence_pruning() -> None:
    answer = (
        "完整机制还包括：\n"
        "1. **空间自适应**：采样密度随区域重要性变化。\n\n"
        "这也是它与简单放大的区别。"
    )

    collapsed = finalize_runtime._collapse_single_item_numbered_blocks(answer)

    assert "完整机制还包括： **空间自适应**：采样密度随区域重要性变化。" in collapsed
    assert "\n1." not in collapsed


def test_collapse_single_item_numbered_block_preserves_real_multi_item_list() -> None:
    answer = "步骤：\n1. 采样。\n2. 重建。"

    assert finalize_runtime._collapse_single_item_numbered_blocks(answer) == answer


def test_sanitize_answer_drops_only_incomplete_markdown_table_tail() -> None:
    answer = (
        "两类选择分别决定编码基底与空间采样几何。\n\n"
        "### 设计层级\n\n"
        "| 决策 | 层级 |\n"
        "| --- | --- |\n"
        "| Hadamard / Fourier | 基底层 |\n"
        "| Foveated supersampling | 空间几何（sp"
    )

    cleaned = finalize_runtime._sanitize_empty_markdown_label_fragments(answer)

    assert cleaned == "两类选择分别决定编码基底与空间采样几何。"


def test_sanitize_answer_removes_empty_citation_attribution_phrase() -> None:
    answer = "如 所述：像素几何由每帧掩模图案定义。"

    cleaned = finalize_runtime._sanitize_empty_markdown_label_fragments(answer)

    assert cleaned == "像素几何由每帧掩模图案定义。"


def test_sanitize_answer_repairs_empty_prior_work_attribution() -> None:
    answer = "作者遵循 的观察，即近端算子可以被其他算子替代。"

    cleaned = finalize_runtime._sanitize_empty_markdown_label_fragments(answer)

    assert cleaned == "作者基于已有工作的观察，即近端算子可以被其他算子替代。"


def test_sanitize_answer_removes_orphan_citation_only_line() -> None:
    answer = "[1]\n\n在该配置下，测得的横向分辨率约为 120 nm [1]。"

    cleaned = finalize_runtime._sanitize_empty_markdown_label_fragments(answer)

    assert cleaned == "在该配置下，测得的横向分辨率约为 120 nm [1]。"


def test_sanitize_answer_removes_only_empty_display_math_blocks() -> None:
    answer = "第一段 [1]。\n\n$$\n$$\n\n$$\nx = Ay\n$$\n\n第二段 [2]。"

    cleaned = finalize_runtime._sanitize_empty_markdown_label_fragments(answer)

    assert "$$\n$$" not in cleaned
    assert "$$\nx = Ay\n$$" in cleaned
    assert cleaned.startswith("第一段 [1]。")
    assert cleaned.endswith("第二段 [2]。")


def test_collapse_duplicate_numeric_citation_across_sentence_punctuation() -> None:
    assert (
        finalize_runtime._collapse_adjacent_duplicate_numeric_citations(
            "该结论由同一证据支持 [1]。[1]"
        )
        == "该结论由同一证据支持 [1]。"
    )


def test_strict_comparison_numbers_use_only_budgeted_system_a_slots() -> None:
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {"preferred_system": "system_a", "candidate_hits": [4]},
            {"preferred_system": "system_a", "candidate_hits": [1]},
            {"preferred_system": "system_a", "candidate_hits": [6]},
        ],
    }

    assert finalize_runtime._strict_comparison_system_a_numbers(plan) == {1, 4}
    assert finalize_runtime._strict_comparison_system_a_numbers(
        {**plan, "intent": "answer_grounding"}
    ) is None


def test_strict_comparison_numbers_do_not_exact_allowlist_one_identified_paper() -> None:
    same_paper_plan = {
        "intent": "comparison",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": "paper.en.md",
                "candidate_hits": [1],
            },
            {
                "preferred_system": "system_a",
                "source_path": "paper.en.md",
                "candidate_hits": [4],
            },
        ],
    }

    assert finalize_runtime._strict_comparison_system_a_numbers(same_paper_plan) is None


def test_plan_hit_resolution_keeps_explicit_candidate_on_same_source_quote_tie() -> None:
    source_path = "qclfm.en.md"
    exact = (
        "Each degree of freedom is measured on separate cameras, and the reported "
        "DOF is 2–5 times larger at 5 μm resolution."
    )
    hits = [
        {
            "text": exact,
            "meta": {"source_path": source_path, "heading_path": "Discussion"},
        },
        {
            "text": "An unrelated setup paragraph.",
            "meta": {"source_path": source_path, "heading_path": "Setup"},
        },
        {
            "text": exact,
            "meta": {"source_path": source_path, "heading_path": "Discussion"},
        },
    ]
    slot = {
        "preferred_system": "system_a",
        "source_path": source_path,
        "heading_path": "Discussion",
        "candidate_hits": [3],
        "evidence_quote": exact,
    }

    assert finalize_runtime._citation_plan_slot_hit_numbers(slot, hits) == [3]


def test_plan_hit_resolution_replaces_low_coverage_stale_candidate() -> None:
    source_path = "paper.en.md"
    exact = (
        "The exact result reports 120 nm resolution at tenfold lower illumination "
        "power, which significantly reduces photodamage."
    )
    hits = [
        {
            "text": "The paper studies resolution and imaging.",
            "meta": {"source_path": source_path, "heading_path": "Abstract"},
        },
        {
            "text": exact,
            "meta": {"source_path": source_path, "heading_path": "Results"},
        },
    ]
    slot = {
        "preferred_system": "system_a",
        "source_path": source_path,
        "heading_path": "Results",
        "candidate_hits": [1],
        "evidence_quote": exact,
    }

    assert finalize_runtime._citation_plan_slot_hit_numbers(slot, hits) == [2]


def test_planned_source_binder_rebinds_reordered_sources_without_adding_prose() -> None:
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_name": "AlphaNet",
                "source_path": "F:/kb/alpha/alpha-net.en.md",
                "candidate_hits": [1],
                "evidence_quote": (
                    "AlphaNet parallelizes coded detector acquisition within one "
                    "integration time."
                ),
            },
            {
                "preferred_system": "system_a",
                "source_name": "BetaGS",
                "source_path": "F:/kb/beta/beta-gs.en.md",
                "candidate_hits": [2],
                "evidence_quote": (
                    "BetaGS reconstructs explicit dynamic 3D scenes from a single "
                    "compressed image."
                ),
            },
        ],
    }
    hits = [
        {
            "text": "A broad BetaGS passage.",
            "meta": {"source_path": "kb/beta/beta-gs.en.md"},
        },
        {
            "text": "A broad AlphaNet passage.",
            "meta": {"source_path": "kb/alpha/alpha-net.en.md"},
        },
    ]
    answer = (
        "AlphaNet parallelizes coded detector acquisition within one integration time.\n\n"
        "BetaGS reconstructs explicit dynamic 3D scenes from a single compressed image."
    )

    bound = finalize_runtime._bind_planned_source_citations(
        answer,
        citation_plan=plan,
        answer_hits=hits,
    )

    assert "integration time [2]." in bound
    assert "compressed image [1]." in bound
    assert bound.replace(" [1]", "").replace(" [2]", "") == answer
    assert (
        finalize_runtime._bind_planned_source_citations(
            bound,
            citation_plan=plan,
            answer_hits=hits,
        )
        == bound
    )


def test_planned_source_binder_places_markers_inside_aligned_table_cells() -> None:
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_name": "SCIGS",
                "source_path": "scigs.en.md",
                "candidate_hits": [1],
                "evidence_quote": (
                    "SCIGS is a variant of 3DGS and reconstructs explicit dynamic 3D "
                    "scenes from a single compressed image."
                ),
            },
            {
                "preferred_system": "system_a",
                "source_name": "BetaGS",
                "source_path": "beta-gs.en.md",
                "candidate_hits": [2],
                "evidence_quote": (
                    "BetaGS reconstructs explicit dynamic 3D scenes from a single "
                    "compressed image."
                ),
            },
        ],
    }
    hits = [
        {"text": "SCIGS source.", "meta": {"source_path": "scigs.en.md"}},
        {"text": "BetaGS source.", "meta": {"source_path": "beta-gs.en.md"}},
    ]
    answer = (
        "| Method | Existing answer claim |\n"
        "| --- | --- |\n"
        "| SCIGS [1] | is a 3DGS variant that reconstructs explicit dynamic 3D scenes from a single compressed image |\n"
        "| BetaGS | reconstructs explicit dynamic 3D scenes from a single compressed image |"
    )

    bound = finalize_runtime._bind_planned_source_citations(
        answer,
        citation_plan=plan,
        answer_hits=hits,
    )

    assert "| Method | Existing answer claim |" in bound
    assert "| --- | --- |" in bound
    assert "| SCIGS [1] | is a 3DGS variant that reconstructs explicit dynamic 3D scenes from a single compressed image [1] |" in bound
    assert "| BetaGS | reconstructs explicit dynamic 3D scenes from a single compressed image [2] |" in bound
    assert "BetaGS [2]" not in bound


def test_planned_source_binder_prefers_complete_taxonomy_over_cited_child_facet() -> None:
    source_path = "review.en.md"
    taxonomy = (
        "Image denoising methods can be roughly classified as spatial domain methods "
        "and transform domain methods."
    )
    spatial_detail = (
        "Spatial domain methods remove noise using correlations between pixels or "
        "image patches."
    )
    plan = {
        "intent": "method_explain",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "heading_path": "Classical denoising method",
                "candidate_hits": [1],
                "evidence_quote": taxonomy,
            },
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "heading_path": "Classical denoising method",
                "candidate_hits": [1],
                "evidence_quote": spatial_detail,
            },
        ],
    }
    hits = [{"text": taxonomy, "meta": {"source_path": source_path}}]
    answer = (
        "这篇综述把经典去噪方法划分为空间域方法和变换域方法两大类。\n\n"
        "空间域方法利用像素或图像块之间的相关性去噪 [1]。"
    )

    bound = finalize_runtime._bind_planned_source_citations(
        answer,
        citation_plan=plan,
        answer_hits=hits,
    )

    assert "两大类 [1]。" in bound
    assert "相关性去噪 [1]。" in bound
    assert (
        finalize_runtime._bind_planned_source_citations(
            bound,
            citation_plan=plan,
            answer_hits=hits,
        )
        == bound
    )


def test_planned_source_binder_leaves_ambiguous_multi_source_claim_unchanged() -> None:
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": "alpha.en.md",
                "candidate_hits": [1],
                "evidence_quote": (
                    "AlphaNet improves reconstruction quality through adaptive sampling."
                ),
            },
            {
                "preferred_system": "system_a",
                "source_path": "beta.en.md",
                "candidate_hits": [2],
                "evidence_quote": (
                    "BetaNet improves reconstruction quality through adaptive sampling."
                ),
            },
        ],
    }
    hits = [
        {"text": "Alpha source.", "meta": {"source_path": "alpha.en.md"}},
        {"text": "Beta source.", "meta": {"source_path": "beta.en.md"}},
    ]
    answer = "The method improves reconstruction quality through adaptive sampling."

    assert (
        finalize_runtime._bind_planned_source_citations(
            answer,
            citation_plan=plan,
            answer_hits=hits,
        )
        == answer
    )


def test_planned_source_binder_adds_correct_marker_before_strict_audit_removes_stale_one() -> None:
    alpha_evidence = (
        "AlphaNet parallelizes coded detector acquisition within one integration time."
    )
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_name": "AlphaNet",
                "source_path": "alpha.en.md",
                "candidate_hits": [1],
                "evidence_quote": alpha_evidence,
            },
            {
                "preferred_system": "system_a",
                "source_name": "BetaNet",
                "source_path": "beta.en.md",
                "candidate_hits": [3],
                "evidence_quote": "BetaNet uses a separate reconstruction strategy.",
            },
        ],
    }
    hits = [
        {
            "text": "An unrelated detector calibration passage.",
            "meta": {"source_path": "other.en.md"},
        },
        {"text": alpha_evidence, "meta": {"source_path": "alpha.en.md"}},
        {
            "text": "BetaNet uses a separate reconstruction strategy.",
            "meta": {"source_path": "beta.en.md"},
        },
    ]
    answer = (
        "AlphaNet parallelizes coded detector acquisition within one integration time [1]."
    )

    bound = finalize_runtime._bind_planned_source_citations(
        answer,
        citation_plan=plan,
        answer_hits=hits,
    )
    repaired, audit = finalize_runtime.audit_and_repair_claim_evidence(
        bound,
        answer_hits=finalize_runtime._claim_evidence_hits_with_citation_plan(hits, plan),
        allowed_citation_numbers=finalize_runtime._strict_comparison_system_a_numbers(
            plan,
            hits,
        ),
        drop_unsupported_unplanned_claims=True,
        enforce_user_visible_binding=True,
    )

    assert "integration time [1][2]." in bound
    assert "integration time [2]." in repaired
    assert "[1]" not in repaired
    assert audit["minimum_ok"] is True


def test_claim_audit_uses_prompt_aligned_plan_evidence_for_numeric_claim() -> None:
    hits = [
        {"text": "Frequency division parallelizes acquisition."},
        {},
        {},
        {
            "text": (
                "Photometric stereo estimates shape from images under different "
                "illumination directions."
            )
        },
    ]
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [4],
                "evidence_quote": (
                    "Using four spatially-separated, single-pixel detectors, the system "
                    "reconstructs continuous real-time 3D video at 8 frames per second."
                ),
            },
            {
                "preferred_system": "system_a",
                "candidate_hits": [1],
                "evidence_quote": "Frequency division parallelizes acquisition.",
            },
        ],
    }

    merged = finalize_runtime._claim_evidence_hits_with_citation_plan(hits, plan)
    answer, audit = finalize_runtime.audit_and_repair_claim_evidence(
        "3D single-pixel video uses four detectors and reaches 8 frames per second [4].",
        answer_hits=merged,
        allowed_citation_numbers={1, 4},
        drop_unsupported_unplanned_claims=True,
    )

    assert "8 frames per second [4]" in answer
    assert audit["dropped_hard_mismatch_claims"] == 0
    assert audit["minimum_ok"] is True


def test_claim_audit_keeps_same_source_deepread_evidence_off_primary_locator() -> None:
    abstract = "PatchTST is an efficient Transformer model for time series forecasting."
    deepread = (
        "PatchTST uses channel-independence, where each channel is a single "
        "univariate time series and shares the same embedding and Transformer weights."
    )
    hits = [
        {
            "text": "A broad PatchTST retrieval passage.",
            "meta": {
                "source_name": "PatchTST.pdf",
                "source_path": "patchtst.en.md",
                "heading_path": "PatchTST / Abstract",
            },
        }
    ]
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 1, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_name": "PatchTST.pdf",
                "source_path": "patchtst.en.md",
                "heading_path": "PatchTST / Abstract",
                "candidate_hits": [1],
                "evidence_quote": abstract,
                "source_evidence_quotes": [deepread],
            }
        ],
    }

    merged = finalize_runtime._claim_evidence_hits_with_citation_plan(hits, plan)

    assert merged[0]["text"] == abstract
    assert merged[0]["meta"]["primary_evidence"]["snippet"] == abstract
    assert merged[0]["meta"]["citation_plan_evidence_quotes"] == [abstract, deepread]
    assert deepread in merged[0]["meta"]["citation_plan_full_evidence_quote"]


def test_pidl_pascal_claim_rebinds_from_adjacent_review_to_primary_paper() -> None:
    pidl_path = "F:/kb/db/pidl/pidl.en.md"
    evidence = (
        "We established a real-world physical noise model of SPAD arrays and calibrated "
        "it with real-shot images. With the calibrated physical noise model, public "
        "high-resolution images from PASCAL VOC2007 and VOC2012 were used to digitally "
        "synthesize 2.6 million image pairs. The gated fusion transformer network was "
        "trained using the large-scale single-photon image dataset."
    )
    hits = [
        {
            "text": "A compact SPAD noise-model passage.",
            "meta": {"source_path": "kb-source/0/pidl/pidl.en.md"},
        },
        {"text": "A detector review.", "meta": {"source_path": "detector.en.md"}},
        {
            "text": "A deep-learning single-pixel imaging review.",
            "meta": {"source_path": "lpr-review.en.md"},
        },
    ]
    plan = {
        "intent": "method_explain",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [1],
                "source_path": pidl_path,
                "heading_path": "Introduction",
                "evidence_quote": evidence,
            }
        ],
    }
    raw = (
        "该方法利用标定的 SPAD 物理噪声模型和 PASCAL VOC2007 图像"
        "合成配对训练数据 [3]。"
    )

    bound = finalize_runtime._bind_planned_source_citations(
        raw,
        citation_plan=plan,
        answer_hits=hits,
    )
    merged = finalize_runtime._claim_evidence_hits_with_citation_plan(hits, plan)
    repaired, audit = finalize_runtime.audit_and_repair_claim_evidence(
        bound,
        answer_hits=merged,
        allow_citation_repairs=True,
        drop_unsupported_high_risk_claims=True,
        enforce_user_visible_binding=True,
    )

    assert "[1]" in repaired
    assert "[3]" not in repaired
    assert "PASCAL VOC2007" in repaired
    assert audit["minimum_ok"] is True


def test_grounded_system_a_allowlist_excludes_unplanned_retrieval_hits() -> None:
    source_path = "F:/kb/db/pidl/pidl.en.md"
    hits = [
        {"text": "planned evidence", "meta": {"source_path": source_path}},
        {"text": "neighboring paragraph", "meta": {"source_path": source_path}},
        {"text": "unplanned review", "meta": {"source_path": "F:/kb/db/review.en.md"}},
    ]
    plan = {
        "budget": {"system_a": 1, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [1],
                "source_path": source_path,
                "evidence_quote": "planned evidence",
            }
        ],
    }

    assert finalize_runtime._planned_grounded_system_a_numbers(plan, hits) == {1}


def test_lora_freeze_normalizer_keeps_source_level_pretrained_weights_term() -> None:
    evidence = (
        "We propose Low-Rank Adaptation, which freezes the pre-trained model weights "
        "and injects trainable rank decomposition matrices."
    )
    answer = (
        "LoRA freezes the pre-trained model weights and injects trainable low-rank "
        "decomposition matrices [1]."
    )

    normalized = finalize_runtime._normalize_citation_plan_supported_terms(
        answer,
        prompt="What does LoRA freeze, and what trainable objects does it inject?",
        citation_plan={
            "slots": [
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [1],
                    "source_path": "lora.en.md",
                    "evidence_quote": evidence,
                }
            ]
        },
        answer_hits=[
            {"text": evidence, "meta": {"source_path": "lora.en.md"}}
        ],
    )

    assert "pre-trained weights" in normalized
    assert "original model weight matrices" in normalized
    assert normalized.count("pre-trained weights") == 1


def test_nerf_position_observation_normalizer_keeps_exact_requested_scope() -> None:
    evidence = (
        "We found that having the network F_Theta directly operate on xyz theta phi "
        "input coordinates results in renderings that perform poorly at representing "
        "high-frequency variation in color and geometry. They additionally show that "
        "mapping the inputs to a higher dimensional space using high frequency functions "
        "before passing them to the network enables better fitting of data that contains "
        "high frequency variation."
    )
    raw = (
        "NeRF uses positional encoding [1].\n\n"
        "An unrelated ablation claim says it is the largest contributor [1]."
    )

    normalized = finalize_runtime._normalize_citation_plan_supported_terms(
        raw,
        prompt=(
            "NeRF 为什么要对输入坐标做 positional encoding？直接把 xyzθφ 输入 MLP 时，"
            "论文观察到的表示问题是什么？"
        ),
        citation_plan={
            "slots": [
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [1],
                    "source_path": "nerf.en.md",
                    "evidence_quote": evidence,
                }
            ]
        },
        answer_hits=[
            {"text": evidence, "meta": {"source_path": "nerf.en.md"}}
        ],
    )

    assert "颜色和几何中的高频变化" in normalized
    assert "high frequency functions" in normalized
    assert "largest contributor" not in normalized
    assert normalized.count("[1]") == 1

    rebound = finalize_runtime._bind_planned_source_citations(
        normalized,
        citation_plan={
            "slots": [
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [1],
                    "source_path": "nerf.en.md",
                    "evidence_quote": evidence,
                }
            ]
        },
        answer_hits=[{"text": evidence, "meta": {"source_path": "nerf.en.md"}}],
    )
    repaired, audit = finalize_runtime.audit_and_repair_claim_evidence(
        rebound,
        answer_hits=[{"text": evidence, "meta": {"source_path": "nerf.en.md"}}],
        allow_citation_repairs=True,
        prompt=(
            "NeRF 为什么要对输入坐标做 positional encoding？直接把 xyzθφ 输入 MLP 时，"
            "论文观察到的表示问题是什么？"
        ),
        allowed_citation_numbers={1},
        drop_unsupported_high_risk_claims=True,
        enforce_user_visible_binding=True,
    )

    assert repaired.count("[1]") == 1
    assert audit["minimum_ok"] is True


def test_planned_binder_cites_each_same_source_module_claim() -> None:
    source_path = "restormer.en.md"
    mdta_evidence = (
        "MDTA applies self-attention across the feature dimension rather than the "
        "spatial dimension and computes cross-covariance across feature channels."
    )
    gdfn_evidence = (
        "where element-wise multiplication and GELU define the gating operation. "
        "Overall, the GDFN controls the information flow through the respective "
        "hierarchical levels, thereby allowing each level to focus on the fine details "
        "complimentary to the other levels. That is, GDFN offers a distinct role "
        "compared to MDTA. Since GDFN performs more operations than the regular feed-"
        "forward network, the expansion ratio is reduced to keep parameters and compute "
        "burden similar."
    )
    plan = {
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [1],
                "source_path": source_path,
                "evidence_quote": mdta_evidence,
            },
            {
                "preferred_system": "system_a",
                "candidate_hits": [1],
                "source_path": source_path,
                "evidence_quote": gdfn_evidence,
            },
        ],
    }
    hits = [{"text": "bundled source", "meta": {"source_path": source_path}}]
    answer = (
        "MDTA applies attention across feature channels instead of spatial positions [1].\n\n"
        "**GDFN:** [1]\n"
        "GDFN controls information flow through the hierarchical levels, allowing each "
        "level to focus on fine details complementary to the other levels. Its gating "
        "role is distinct from MDTA. Because GDFN performs more operations than a regular "
        "feed-forward network, Restormer reduces the expansion ratio to keep parameters and "
        "compute burden similar [1]."
    )

    bound = finalize_runtime._bind_planned_source_citations(
        answer,
        citation_plan=plan,
        answer_hits=hits,
    )

    assert "fine details complementary to the other levels [1]" in bound
    merged = finalize_runtime._claim_evidence_hits_with_citation_plan(hits, plan)
    repaired, audit = finalize_runtime.audit_and_repair_claim_evidence(
        bound,
        answer_hits=merged,
        allow_citation_repairs=True,
        allowed_citation_numbers={1},
        drop_unsupported_high_risk_claims=True,
        enforce_user_visible_binding=True,
    )

    assert "fine details complementary to the other levels [1]" in repaired
    assert audit["minimum_ok"] is True


def test_supported_term_normalizer_keeps_gdfn_gating_in_cited_mechanism() -> None:
    evidence = (
        "Overall, the GDFN controls the information flow through the hierarchical "
        "levels, allowing each level to focus on fine details complementary to others."
    )
    answer = (
        "While MDTA enriches contextual information, GDFN regulates the information "
        "flow through each hierarchical level and preserves fine details [1]."
    )

    normalized = finalize_runtime._normalize_citation_plan_supported_terms(
        answer,
        prompt=(
            "In Restormer, what does MDTA transpose and what distinct filtering role "
            "does GDFN play?"
        ),
        citation_plan={
            "slots": [
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [1],
                    "source_path": "restormer.en.md",
                    "evidence_quote": evidence,
                }
            ]
        },
        answer_hits=[
            {"text": evidence, "meta": {"source_path": "restormer.en.md"}}
        ],
    )

    assert "GDFN gates and regulates the information flow" in normalized
    assert normalized.endswith("[1].")


def test_preflight_source_slot_resolves_against_final_hit_order() -> None:
    source_path = "F:/blind/md/sam/sam.en.md"
    plan = {
        "budget": {"system_a": 1, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [],
                "source_path": source_path,
                "evidence_quote": (
                    "An image encoder computes an image embedding, a prompt encoder embeds "
                    "prompts, and a mask decoder predicts segmentation masks."
                ),
            }
        ],
    }

    resolved = finalize_runtime._citation_plan_with_resolved_hit_numbers(
        plan,
        answer_hits=[
            {
                "text": "SAM source evidence",
                "meta": {"source_path": "kb-source/0/sam/sam.en.md"},
            }
        ],
    )

    assert resolved["slots"][0]["candidate_hits"] == [1]
    assert plan["slots"][0]["candidate_hits"] == []


def test_pidl_pascal_normalizer_rewrites_only_the_unsupported_disentanglement_tail() -> None:
    source_path = "F:/kb/db/High-resolution single-photon imaging with physics-informed deep learning.en.md"
    evidence = (
        "With the calibrated physical noise model under different illumination and acquisition "
        "settings, public highresolution images collected from the PASCAL VOC2007 and VOC2012 "
        "datasets were used to digitally synthesize a large-scale realistic singlephoton image "
        "dataset containing 2.6 million image pairs. The gated fusion transformer network was "
        "trained using the above large-scale singlephoton image dataset."
    )
    plan = {
        "intent": "method_explain",
        "budget": {"system_a": 1, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [1],
                "source_path": source_path,
                "source_name": "High-resolution single-photon imaging with physics-informed deep learning",
                "heading_path": "Introduction",
                "evidence_quote": evidence,
            }
        ],
    }
    raw = (
        "该方法先用真实 SPAD 图像标定物理噪声模型 [1]。然后，利用这个标定好的物理模型，"
        "结合公开的高分辨率自然图像（如 PASCAL VOC2007）生成大量配对的有噪-清晰训练数据，"
        "使网络学会从物理噪声中解耦出真实信号 [3]。"
    )

    normalized = finalize_runtime._normalize_citation_plan_supported_terms(
        raw,
        prompt="physics-informed deep learning 在单光子成像中的核心作用是什么？",
        citation_plan=plan,
        answer_hits=[
            {"text": evidence, "meta": {"source_path": source_path}},
            {"text": "A detector review.", "meta": {"source_path": "review.en.md"}},
            {"text": "A broad SPI review.", "meta": {"source_path": "lpr.en.md"}},
        ],
    )

    assert "从物理噪声中解耦出真实信号" not in normalized
    assert "有噪-清晰训练数据" not in normalized
    assert (
        "利用标定后的物理噪声模型和 PASCAL VOC2007/VOC2012 公共高分辨率图像，"
        "数字合成大规模真实单光子图像数据集，并用该数据集训练网络 [1]。"
    ) in normalized
    assert "先用真实 SPAD 图像标定物理噪声模型 [1]" in normalized


def test_pidl_pascal_normalizer_does_not_rewrite_without_full_training_contract() -> None:
    source_path = "F:/kb/db/High-resolution single-photon imaging with physics-informed deep learning.en.md"
    incomplete_evidence = (
        "With the calibrated physical noise model, public images from PASCAL VOC2007 were "
        "used for a single-photon imaging experiment."
    )
    raw = (
        "利用这个标定好的物理模型和 PASCAL VOC2007 图像生成训练数据，"
        "使网络学会从物理噪声中解耦出真实信号 [1]。"
    )

    normalized = finalize_runtime._normalize_citation_plan_supported_terms(
        raw,
        prompt="介绍该方法。",
        citation_plan={
            "budget": {"system_a": 1, "system_b": 0},
            "slots": [
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [1],
                    "source_path": source_path,
                    "source_name": "High-resolution single-photon imaging with physics-informed deep learning",
                    "evidence_quote": incomplete_evidence,
                }
            ],
        },
        answer_hits=[
            {"text": incomplete_evidence, "meta": {"source_path": source_path}}
        ],
    )

    assert normalized == raw


def test_pidl_pascal_normalizer_removes_unsupported_reconstruction_result_tail() -> None:
    source_path = "F:/kb/db/High-resolution single-photon imaging with physics-informed deep learning.en.md"
    evidence = (
        "With the calibrated physical noise model, public highresolution images from "
        "PASCAL VOC2007 and VOC2012 were used to digitally synthesize a large-scale "
        "realistic singlephoton image dataset containing 2.6 million image pairs. "
        "The gated fusion transformer network was trained using the above dataset."
    )
    raw = (
        "最后，利用校准后的模型和公开的高分辨率图像（如 PASCAL VOC2007）合成配对数据，"
        "用于训练网络，从而实现对低质量 SPAD 图像的高分辨率重建 [1]。"
    )

    normalized = finalize_runtime._normalize_citation_plan_supported_terms(
        raw,
        prompt="这篇 physics-informed deep learning 论文怎样训练网络？",
        citation_plan={
            "budget": {"system_a": 1, "system_b": 0},
            "slots": [
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [1],
                    "source_path": source_path,
                    "source_name": "High-resolution single-photon imaging with physics-informed deep learning",
                    "evidence_quote": evidence,
                }
            ],
        },
        answer_hits=[{"text": evidence, "meta": {"source_path": source_path}}],
    )

    assert "从而实现" not in normalized
    assert "高分辨率重建" not in normalized
    assert "PASCAL VOC2007/VOC2012" in normalized
    assert "训练网络 [1]" in normalized


def test_detector_pidl_reading_pair_restores_both_grounded_sources() -> None:
    review_path = "F:/kb/db/spd-review/spd-review.en.md"
    pidl_path = "F:/kb/db/pidl/pidl.en.md"
    review_evidence = (
        "This technology mainly relies on the mainstream SPDs, such as photomultiplier "
        "tubes (PMTs), avalanche photodiodes (SAPD), superconducting nanowire "
        "single-photon detectors (SNSPDs), and superconducting transition-edge sensor "
        "(TES). High manufacturing cost and special conditions like a low-temperature "
        "environment pose challenges to adoption."
    )
    pidl_evidence = (
        "We established a real-world physical noise model of SPAD arrays. With the "
        "calibrated physical noise model, public images from PASCAL VOC2007 and VOC2012 "
        "were used to digitally synthesize a realistic single-photon image dataset."
    )
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": review_path,
                "candidate_hits": [2],
                "evidence_quote": review_evidence,
            },
            {
                "preferred_system": "system_a",
                "source_path": pidl_path,
                "candidate_hits": [1],
                "evidence_quote": pidl_evidence,
            },
        ],
    }
    raw = (
        "### 1. 先读：探测器综述——建立硬件基础\n\n"
        "- 这能让你对SPAD阵列的硬件局限有直观认识，为理解下一篇的噪声模型打下基础 [1]。\n\n"
        "### 2. 后读：物理信息深度学习\n\n"
        "- **关键区别**：这篇论文不是黑盒深度学习，而是把先验注入网络。\n\n"
        "### 搭配阅读建议\n\n"
        "**注意模态边界**：讨论的是单像素成像，不要混淆。和才是配套材料。"
    )

    normalized = finalize_runtime._normalize_citation_plan_supported_terms(
        raw,
        prompt="单光子成像里，探测器综述和 physics-informed deep learning 这篇应该怎么搭配读？",
        citation_plan=plan,
        answer_hits=[
            {"text": pidl_evidence, "meta": {"source_path": "kb-source/0/pidl/pidl.en.md"}},
            {"text": review_evidence, "meta": {"source_path": "kb-source/0/spd-review/spd-review.en.md"}},
        ],
    )

    assert "PMT、SAPD、SNSPD、TES" in normalized
    assert "特殊工作条件会限制普及 [2]" in normalized
    assert "训练数据合成流程 [1]" in normalized
    assert "不是黑盒" not in normalized
    assert "注意模态边界" not in normalized
    assert "和才是配套材料" not in normalized


def test_piln_review_positioning_uses_review_definition_and_removes_overreach() -> None:
    piln_path = "F:/kb/db/piln/piln.en.md"
    review_path = "F:/kb/db/dl-spi-review/dl-spi-review.en.md"
    piln_evidence = (
        "We proposed a self-supervised image-loop neural network (ILNet). 1D signals "
        "collected by the single-pixel detector are used as labels for adaptively "
        "optimizing and reconstructing the image. The method works at lower sample "
        "rates in unknown free-space and underwater experiments."
    )
    review_evidence = (
        "Model-driven strategy is an unsupervised learning mode. This strategy "
        "integrates the physical process of SPI with neural networks and leverages "
        "the discrepancy between real and estimated measurements to guide network optimization."
    )
    plan = {
        "intent": "scope_boundary",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": piln_path,
                "candidate_hits": [1],
                "evidence_quote": piln_evidence,
            },
            {
                "preferred_system": "system_a",
                "source_path": review_path,
                "candidate_hits": [2],
                "evidence_quote": review_evidence,
            },
        ],
    }
    raw = (
        "### 1. 定位：一种自监督、混合驱动的具体实现\n\n"
        "PILN（ILNet）属于 DL-SPI 中的混合驱动策略。\n\n"
        "其核心数学表达为：\n\n$$\nimage=f(x)\n$$\n\n$$\n$$\n\n"
        "### 2. 适合解决的问题\n\n低采样率重建 [1]。\n\n"
        "### 3. 不适合/未解决的问题\n\n"
        "| 场景 | 说明 |\n|---|---|\n| 理论保证 | 作为纯神经网络缺乏理论保证 |\n\n"
        "### 4. 与主线的本质关系\n\n"
        "ILNet 是数据驱动和模型驱动融合的案例。"
    )

    normalized = finalize_runtime._normalize_citation_plan_supported_terms(
        raw,
        prompt="PILN 这种网络方法和综述里说的深度学习单像素成像主线是什么关系？",
        citation_plan=plan,
        answer_hits=[
            {"text": piln_evidence, "meta": {"source_path": "kb-source/0/piln/piln.en.md"}},
            {"text": review_evidence, "meta": {"source_path": "kb-source/0/dl-spi-review/dl-spi-review.en.md"}},
        ],
    )

    assert "model-driven strategy 定义为一种无监督模式" in normalized
    assert "差异指导优化 [2]" in normalized
    assert "一维信号作为标签" in normalized
    assert "重建图像 [1]" in normalized
    assert "混合驱动策略" not in normalized
    assert "纯神经网络" not in normalized
    assert "核心数学表达" not in normalized
    assert "photon-level、实时吞吐量或理论收敛保证" in normalized


def test_piln_review_positioning_rebuilds_complete_exact_scope_answer() -> None:
    piln_evidence = (
        "We proposed a self-supervised image-loop neural network (ILNet) with a "
        "part-based model for single-pixel imaging. The part-based model divides image "
        "features into different parts to facilitate finer-grained learning. 1D signals "
        "collected by the single-pixel detector are used as labels for adaptively "
        "optimizing and reconstructing the image. ILNet reconstructs high-quality images "
        "with lower sample rates in unknown free-space and underwater experiments."
    )
    review_evidence = (
        "Model-driven strategy is an unsupervised learning mode that exhibits exceptional "
        "generalization. It integrates the physical process of SPI with neural networks "
        "and leverages the discrepancy between real and estimated measurements to guide "
        "network optimization."
    )
    normalized = finalize_runtime._normalize_citation_plan_supported_terms(
        "PILN 是混合驱动方法。\n\n$$\n$$\n\n它什么都能做 [3]。",
        prompt=(
            "PILN 这种网络方法和综述里说的深度学习单像素成像主线是什么关系？"
            "它适合解决什么，不适合解决什么？"
        ),
        citation_plan={
            "intent": "scope_boundary",
            "budget": {"system_a": 2, "system_b": 0},
            "slots": [
                {
                    "preferred_system": "system_a",
                    "source_path": "piln.en.md",
                    "candidate_hits": [1],
                    "evidence_quote": piln_evidence,
                },
                {
                    "preferred_system": "system_a",
                    "source_path": "review.en.md",
                    "candidate_hits": [2],
                    "evidence_quote": review_evidence,
                },
            ],
        },
        answer_hits=[
            {"text": piln_evidence, "meta": {"source_path": "piln.en.md"}},
            {"text": review_evidence, "meta": {"source_path": "review.en.md"}},
        ],
    )

    assert "part-based image-loop network" in normalized
    assert "finer-grained learning" in normalized
    assert "generalization" in normalized
    assert "未知自由空间和水下实验 [1]" in normalized
    assert "实时吞吐量" in normalized
    assert "$$" not in normalized
    assert "[3]" not in normalized


def test_cross_paper_plan_resolves_reordered_hits_by_source_before_evidence_audit() -> None:
    private_paths = {
        "alpha": "F:/kb/db/alpha/alpha.en.md",
        "beta": "F:/kb/db/beta/beta.en.md",
        "gamma": "F:/kb/db/gamma/gamma.en.md",
    }
    plan = {
        "intent": "answer_grounding",
        "budget": {"system_a": 3, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": private_paths["alpha"],
                "heading_path": "Abstract",
                "evidence_quote": "Paper Alpha establishes the coded measurement model.",
                "candidate_hits": [1],
            },
            {
                "preferred_system": "system_a",
                "source_path": private_paths["beta"],
                "heading_path": "Methods",
                "evidence_quote": "Paper Beta parallelizes hardware acquisition.",
                "candidate_hits": [2],
            },
            {
                "preferred_system": "system_a",
                "source_path": private_paths["gamma"],
                "heading_path": "Results",
                "evidence_quote": "Paper Gamma adds physics-informed reconstruction.",
                "candidate_hits": [3],
            },
        ],
    }
    # Public answer hits were reranked after the plan recorded private-path
    # candidate numbers; an unplanned paper also entered the visible window.
    hits = [
        {
            "text": "A broad Beta passage.",
            "meta": {"source_path": "kb-source/0/beta/beta.en.md"},
        },
        {
            "text": "Paper Delta discusses an unrelated detector.",
            "meta": {"source_path": "kb-source/0/delta/delta.en.md"},
        },
        {
            "text": "A broad Alpha passage.",
            "meta": {"source_path": "kb-source/0/alpha/alpha.en.md"},
        },
        {
            "text": "A broad Gamma passage.",
            "meta": {"source_path": "kb-source/0/gamma/gamma.en.md"},
        },
    ]

    allowed = finalize_runtime._strict_comparison_system_a_numbers(plan, hits)
    merged = finalize_runtime._claim_evidence_hits_with_citation_plan(hits, plan)

    assert allowed == {1, 3, 4}
    assert merged[0]["text"] == "Paper Beta parallelizes hardware acquisition."
    assert merged[1]["text"] == "Paper Delta discusses an unrelated detector."
    assert merged[2]["text"] == "Paper Alpha establishes the coded measurement model."
    assert merged[3]["text"] == "Paper Gamma adds physics-informed reconstruction."


def test_cross_paper_claims_rebind_to_same_sources_as_reference_cards() -> None:
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": "F:/kb/db/fdm/fdm.en.md",
                "evidence_quote": (
                    "Frequency-division multiplexing parallelizes multiple patterns "
                    "within one detector integration time."
                ),
                "candidate_hits": [1],
            },
            {
                "preferred_system": "system_a",
                "source_path": "F:/kb/db/video-3d/video-3d.en.md",
                "evidence_quote": (
                    "Photometric stereo uses four spatially separated detectors "
                    "for parallel directional measurements."
                ),
                "candidate_hits": [2],
            },
        ],
    }
    hits = [
        {
            "text": "A broad 3D video passage.",
            "meta": {"source_path": "kb-source/0/video-3d/video-3d.en.md"},
        },
        {
            "text": "An unrelated review passage.",
            "meta": {"source_path": "kb-source/0/review/review.en.md"},
        },
        {
            "text": "A broad frequency-division passage.",
            "meta": {"source_path": "kb-source/0/fdm/fdm.en.md"},
        },
    ]
    merged = finalize_runtime._claim_evidence_hits_with_citation_plan(hits, plan)
    allowed = finalize_runtime._strict_comparison_system_a_numbers(plan, hits)

    repaired, audit = finalize_runtime.audit_and_repair_claim_evidence(
        (
            "Frequency-division multiplexing parallelizes multiple patterns within "
            "one detector integration time [1]. "
            "Photometric stereo uses four spatially separated detectors [2]."
        ),
        answer_hits=merged,
        allowed_citation_numbers=allowed,
        drop_unsupported_unplanned_claims=True,
    )

    assert allowed == {1, 3}
    assert "integration time [3]" in repaired
    assert "four spatially separated detectors [1]" in repaired
    assert "[2]" not in repaired
    assert audit["minimum_ok"] is True


def test_fdm_answer_replaces_secondary_result_with_planned_parallel_mechanism() -> None:
    evidence = (
        "Here, we implement frequency-division methods to parallelize the "
        "single-pixel imaging process at 3.2 THz. Our technique enables a "
        "trade-off between signal-to-noise ratio and acquisition speed—without "
        "altering detector integration time."
    )
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 1, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": "F:/kb/db/fdm/fdm.en.md",
                "evidence_quote": evidence,
                "candidate_hits": [1],
            }
        ],
    }
    answer = (
        "频分复用并行化了成像过程。实验表明，该技术实现了四倍的效率提升，"
        "并且这种加速效果对任意图像尺寸都完全可扩展 [3]。\n\n"
        "代价是信噪比（SNR）与采集速度之间的权衡 [1]。\n\n"
        "不需要改变探测器积分时间 [1]。"
    )

    normalized = finalize_runtime._normalize_citation_plan_supported_terms(
        answer,
        prompt="频分复用为什么更快，代价是什么？",
        citation_plan=plan,
        answer_hits=[
            {
                "text": evidence,
                "meta": {"source_path": "kb-source/0/fdm/fdm.en.md"},
            }
        ],
    )

    assert "并行化单像素成像过程" in normalized
    assert "探测器积分时间" in normalized
    assert "[1]" in normalized
    assert "四倍的效率提升" not in normalized
    assert "任意图像尺寸" not in normalized


def test_fdm_comparison_completes_bare_encoding_side_from_planned_mechanism() -> None:
    fdm_evidence = (
        "The mask values are encoded in the phase of intensity modulation, and thus "
        "we require phase-sensitive detection with a lock-in amplifier. Each pixel of "
        "the SLM is modulated with either 0 or pi phase on p frequencies simultaneously. "
        "The modulated light from the SLM is then multiplexed into a single-pixel "
        "detector. The signal is then demodulated by a number (p) of LIAs."
    )
    video_evidence = (
        "Photometric stereo senses reflected light with four spatially-separated "
        "single-pixel detectors and reconstructs real-time 3D video."
    )
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [2],
                "source_path": "video.en.md",
                "evidence_quote": video_evidence,
            },
            {
                "preferred_system": "system_a",
                "candidate_hits": [1],
                "source_path": "Frequency-division-multiplexed single-pixel imaging.en.md",
                "heading_path": "B. Encoding",
                "evidence_quote": fdm_evidence,
            },
        ],
    }
    hits = [
        {
            "text": fdm_evidence,
            "meta": {
                "source_path": "Frequency-division-multiplexed single-pixel imaging.en.md"
            },
        },
        {"text": video_evidence, "meta": {"source_path": "video.en.md"}},
    ]
    answer = (
        "两种方法都追求速度，但并行环节不同 [2]。\n\n"
        "**频分复用单像素成像（FDM-SPI）**并行化调制/编码环节。\n\n"
        "**3D single-pixel video**使用四个探测器并行采集 [2]。"
    )

    normalized = finalize_runtime._normalize_citation_plan_supported_terms(
        answer,
        prompt="两种方法分别把什么环节并行化？",
        citation_plan=plan,
        answer_hits=hits,
    )

    fdm_paragraph = next(
        paragraph for paragraph in normalized.split("\n\n") if "FDM-SPI" in paragraph
    )
    assert "p 个频率通道" in fdm_paragraph
    assert "同一个单像素探测器" in fdm_paragraph
    assert "锁相放大器" in fdm_paragraph
    assert fdm_paragraph.endswith("[1]。")


def test_fdm_completion_does_not_invent_phase_or_lockin_details() -> None:
    evidence = (
        "Each SLM pixel is used to modulate p frequencies simultaneously. "
        "The modulated light is multiplexed into a single-pixel detector. "
        "The signal is then demodulated."
    )
    source_path = "Frequency-division-multiplexed single-pixel imaging.en.md"
    plan = {
        "intent": "method_explain",
        "budget": {"system_a": 1, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [1],
                "source_path": source_path,
                "heading_path": "B. Encoding",
                "evidence_quote": evidence,
            }
        ],
    }

    normalized = finalize_runtime._normalize_citation_plan_supported_terms(
        "FDM-SPI parallelizes its encoding stage.",
        prompt="How does FDM-SPI encode its channels?",
        citation_plan=plan,
        answer_hits=[{"text": evidence, "meta": {"source_path": source_path}}],
    )

    assert "p frequencies simultaneously" in normalized
    assert "one single-pixel detector" in normalized
    assert "then demodulated [1]" in normalized
    assert "0/π" not in normalized
    assert "lock-in" not in normalized
    assert "phase-sensitive" not in normalized
    assert "spatial-mask" not in normalized


def test_final_gate_drops_unstated_pidl_black_box_and_scene_robustness_claim() -> None:
    evidence = (
        "We established a real-world physical noise model of SPAD arrays and calibrated "
        "it with real-shot images. The calibrated model was used to synthesize image "
        "pairs for network training."
    )
    answer = (
        "该方法建立并标定了 SPAD 阵列的真实物理噪声模型 [1]。\n\n"
        "该方法使用物理噪声模型替代纯数据驱动的黑箱学习，"
        "使网络在训练数据有限或场景变化时仍能保持鲁棒性 [1]。"
    )

    repaired, audit = finalize_runtime.audit_and_repair_claim_evidence(
        answer,
        answer_hits=[{"text": evidence}],
        allow_citation_repairs=True,
        allowed_citation_numbers={1},
        drop_unsupported_unplanned_claims=True,
        drop_unsupported_high_risk_claims=True,
        enforce_user_visible_binding=True,
    )

    assert "建立并标定了 SPAD 阵列的真实物理噪声模型 [1]" in repaired
    assert "黑箱" not in repaired
    assert "场景变化" not in repaired
    assert audit["minimum_ok"] is True


def test_four_hit_comparison_uses_plan_source_identity_for_claim_rebinding() -> None:
    scinerf_path = "F:/kb/db/scinerf/scinerf.en.md"
    scigs_path = "F:/kb/db/scigs/scigs.en.md"
    scinerf_quote = (
        "Specifically, we formulate the physical imaging process of SCI as part "
        "of the training of NeRF."
    )
    scigs_quote = (
        "SCIGS is a variant of 3DGS and reconstructs explicit dynamic 3D scenes "
        "from a single compressed image."
    )
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": scinerf_path,
                "source_name": "SCINeRF",
                "heading_path": "SCINeRF / Abstract",
                "evidence_quote": scinerf_quote,
                "candidate_hits": [2],
            },
            {
                "preferred_system": "system_a",
                "source_path": scigs_path,
                "source_name": "SCIGS",
                "heading_path": "SCIGS / Abstract",
                "evidence_quote": scigs_quote,
                "candidate_hits": [4],
            },
        ],
    }
    hits = [
        {"text": "SCIGS title.", "meta": {"source_path": scigs_path}},
        {"text": "SCINeRF title.", "meta": {"source_path": scinerf_path}},
        {
            "text": "An unrelated SCINeRF mask-overlap passage.",
            "meta": {
                "source_path": scinerf_path,
                "heading_path": "SCINeRF / Abstract",
            },
        },
        {
            "text": "SCIGS is a variant of 3DGS. A broad SCIGS comparison table.",
            "meta": {"source_path": scigs_path},
        },
    ]

    merged = finalize_runtime._claim_evidence_hits_with_citation_plan(hits, plan)
    repaired, audit = finalize_runtime.audit_and_repair_claim_evidence(
        (
            "SCIGS reconstructs explicit dynamic 3D scenes from a single compressed "
            "image [3]. SCINeRF formulates the physical imaging process of SCI as "
            "part of the training of NeRF [3]. SCIGS is a variant of 3DGS [4]."
        ),
        answer_hits=merged,
        allowed_citation_numbers={3, 4},
        drop_unsupported_unplanned_claims=True,
    )

    assert merged[2]["meta"]["source_name"] == "SCINeRF"
    assert merged[3]["meta"]["source_name"] == "SCIGS"
    assert "single compressed image [4]" in repaired
    assert "training of NeRF [3]" in repaired
    assert audit["rebound_citations"] >= 1
    assert audit["minimum_ok"] is True


def test_same_paper_method_slot_binds_abstract_once_and_drops_unsupported_detail() -> None:
    abstract = (
        "Specifically, we formulate the physical imaging process of SCI as part "
        "of the training of NeRF, allowing us to capture complex scene structures."
    )
    source_path = "F:/kb/db/scinerf/scinerf.en.md"
    plan = {
        "intent": "method_explain",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "claim_type": "method_detail",
                "preferred_system": "system_a",
                "source_path": source_path,
                "heading_path": "SCINeRF / Abstract",
                "evidence_quote": abstract,
                "candidate_hits": [],
            },
            {
                "claim_type": "paper_evidence",
                "preferred_system": "system_a",
                "source_path": source_path,
                "heading_path": "SCINeRF / Abstract",
                "evidence_quote": abstract,
                "candidate_hits": [1],
            },
            {
                "claim_type": "paper_evidence",
                "preferred_system": "system_a",
                "source_path": source_path,
                "heading_path": "SCINeRF / Abstract",
                "evidence_quote": f"## Abstract {abstract}",
                "candidate_hits": [2],
            },
        ],
    }
    hits = [
        {
            "text": "Table 2 reports novel-view synthesis metrics.",
            "meta": {
                "source_path": source_path,
                "heading_path": "4. Experiments / Table 2",
            },
        },
        {
            "text": f"## Abstract {abstract}",
            "meta": {
                "source_path": source_path,
                "heading_path": "SCINeRF / Abstract",
            },
        },
    ]

    merged = finalize_runtime._claim_evidence_hits_with_citation_plan(hits, plan)

    assert merged[0]["text"] == "Table 2 reports novel-view synthesis metrics."
    assert "physical imaging process of SCI" in merged[1]["text"]
    assert "citation_plan_evidence_quotes" not in merged[0].get("meta", {})
    assert len(merged[1]["meta"]["citation_plan_evidence_quotes"]) == 2
    assert merged[1]["meta"]["heading_path"] == "SCINeRF / Abstract"
    assert "physical imaging process of SCI" in merged[1]["meta"]["evidence_quote"]
    assert merged[1]["meta"]["primary_evidence"]["heading_path"] == "SCINeRF / Abstract"
    assert "physical imaging process of SCI" in merged[1]["ui_meta"]["primary_evidence"]["snippet"]

    repaired, audit = finalize_runtime.audit_and_repair_claim_evidence(
        (
            "SCINeRF \u5c06 SCI \u7684\u7269\u7406\u6210\u50cf\u8fc7\u7a0b\u4f5c\u4e3a NeRF "
            "\u8bad\u7ec3\u7684\u4e00\u90e8\u5206\u3002"
            "SCI \u68af\u5ea6\u901a\u8fc7\u53ef\u5fae\u538b\u7f29\u5c42\u53cd\u5411\u4f20\u64ad\uff0c"
            "\u66f4\u65b0 NeRF \u53c2\u6570\u3002"
            "NeRF \u6e32\u67d3\u5e27\u5148\u7ecf\u8fc7\u63a9\u7801\u8c03\u5236\u548c\u79ef\u5206\u538b\u7f29\uff0c"
            "\u518d\u4e0e\u771f\u5b9e\u6d4b\u91cf\u6bd4\u8f83\u3002"
        ),
        answer_hits=merged,
        drop_unsupported_high_risk_claims=True,
        enforce_user_visible_binding=True,
    )

    assert "\u4f5c\u4e3a NeRF \u8bad\u7ec3\u7684\u4e00\u90e8\u5206 [2]" in repaired
    assert "\u53cd\u5411\u4f20\u64ad" not in repaired
    assert "\u53ef\u5fae\u538b\u7f29\u5c42" not in repaired
    assert "\u63a9\u7801\u8c03\u5236" not in repaired
    assert "\u79ef\u5206\u538b\u7f29" not in repaired
    assert audit["dropped_unsupported_unplanned_claims"] == 2
    assert audit["minimum_ok"] is True


def test_late_evidence_cards_preserve_two_facets_from_one_paper() -> None:
    source_path = "F:/kb/learned-primal-dual/paper.en.md"
    plan = {
        "budget": {"system_a": 2, "system_b": 0},
        "per_paragraph_budget": {"system_a": 1, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "heading_path": "Paper / Abstract",
                "evidence_quote": "A broad abstract about learned reconstruction.",
                "candidate_hits": [1],
            }
        ],
    }
    cards = [
        {
            "source_path": source_path,
            "source_name": "Learned Primal-Dual Reconstruction.pdf",
            "claim_type": "method_detail",
            "primary_evidence": {
                "source_path": source_path,
                "source_name": "Learned Primal-Dual Reconstruction.pdf",
                "heading_path": "B. Learned PDHG",
                "block_id": "blk_method",
                "anchor_id": "p_method",
                "page_start": 4,
                "snippet": (
                    "The primal proximal has been replaced by a learned proximal and "
                    "the dual proximal by a learned proximal."
                ),
            },
        },
        {
            "source_path": source_path,
            "source_name": "Learned Primal-Dual Reconstruction.pdf",
            "claim_type": "method_detail",
            "primary_evidence": {
                "source_path": source_path,
                "source_name": "Learned Primal-Dual Reconstruction.pdf",
                "heading_path": "C. Learned Primal-Dual / Choice of starting point",
                "block_id": "blk_start",
                "anchor_id": "p_start",
                "page_start": 5,
                "snippet": (
                    "The initial guess did not give better final results, while the "
                    "pseudo-inverse added complexity, so results use zero-initialization."
                ),
            },
        },
    ]
    hits = [
        {
            "text": "The learned method unrolls a primal-dual optimization method.",
            "meta": {"source_path": source_path, "heading_path": "Paper / Abstract"},
        }
    ]

    refreshed = finalize_runtime._citation_plan_with_late_evidence_cards(
        plan,
        evidence_cards=cards,
        support_slots=[],
        answer_hits=hits,
        prompt=(
            "\u8bf7\u5206\u4e24\u90e8\u5206\u89e3\u91ca\u53ef\u5b66\u4e60\u66f4\u65b0\u4e0e\u96f6\u521d\u59cb\u5316\uff0c"
            "\u5e76\u5206\u522b\u7ed9\u51fa\u8bc1\u636e\u3002"
        ),
    )

    assert refreshed["late_evidence_refresh"] is True
    assert [slot["block_id"] for slot in refreshed["slots"]] == [
        "blk_method",
        "blk_start",
    ]
    assert all(slot["candidate_hits"] == [1] for slot in refreshed["slots"])
    assert refreshed["per_paragraph_budget"]["system_a"] == 2

    merged = finalize_runtime._claim_evidence_hits_with_citation_plan(hits, refreshed)
    assert "primal proximal" in merged[0]["text"]
    assert "zero-initialization" in merged[0]["text"]

    refreshed_from_scanner_slots = finalize_runtime._citation_plan_with_late_evidence_cards(
        plan,
        evidence_cards=[],
        support_slots=cards,
        answer_hits=hits,
        prompt=(
            "\u8bf7\u5206\u4e24\u90e8\u5206\u89e3\u91ca\u53ef\u5b66\u4e60\u66f4\u65b0\u4e0e\u96f6\u521d\u59cb\u5316\uff0c"
            "\u5e76\u5206\u522b\u7ed9\u51fa\u8bc1\u636e\u3002"
        ),
    )
    assert refreshed_from_scanner_slots["late_evidence_refresh"] is True
    assert [slot["block_id"] for slot in refreshed_from_scanner_slots["slots"]] == [
        "blk_method",
        "blk_start",
    ]


def test_bind_resolved_support_source_citations_keeps_translated_explanations() -> None:
    source_path = "F:/kb/learned-primal-dual/paper.en.md"
    answer = (
        '## 初始化选择\n\n'
        '> "The initial guess marginally decreased training time, but did not give better final results."'
    )
    support_resolution = [
        {
            "segment_kind": "paragraph",
            "segment_text": "理由一（最终效果）：伪逆初值没有改善最终结果。",
            "source_path": source_path,
            "block_id": "blk_start",
            "anchor_id": "p_start",
        },
        {
            "segment_kind": "paragraph",
            "segment_text": "理由二（额外复杂度）：伪逆会增加系统复杂度并依赖先前的重建。",
            "source_path": source_path,
            "block_id": "blk_start",
            "anchor_id": "p_start",
        },
    ]
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": source_path,
                "block_id": "blk_start",
                "candidate_hits": [1],
            }
        ]
    }
    hits = [
        {
            "text": "source evidence",
            # The canonical hit can retain the PDF path while late evidence
            # slots and support records point at the converted Markdown.
            "meta": {"source_path": "F:/library/Learned Primal-Dual Reconstruction.pdf"},
        }
    ]

    bound = finalize_runtime._bind_resolved_support_source_citations(
        answer,
        support_resolution=support_resolution,
        answer_hits=hits,
        citation_plan=plan,
    )

    assert "理由一（最终效果）：伪逆初值没有改善最终结果 [1]。" in bound
    assert "理由二（额外复杂度）：伪逆会增加系统复杂度并依赖先前的重建 [1]。" in bound
    assert '> "The initial guess marginally decreased training time' in bound

def test_merge_citation_plan_support_slots_preserves_prompt_aligned_source_sentence():
    merged = finalize_runtime._merge_citation_plan_support_slots(
        [
            {
                "doc_idx": 1,
                "support_id": "DOC-1",
                "support_example": "[[SUPPORT:DOC-1]]",
                "source_path": "paper.en.md",
                "locate_anchor": "A weak background sentence.",
            }
        ],
        citation_plan={
            "slots": [
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [],
                    "source_path": "paper.en.md",
                    "heading_path": "Paper / Abstract",
                    "evidence_quote": (
                        "The method operates at tenfold lower incident illumination power, "
                        "significantly reducing photodamage."
                    ),
                    "block_id": "blk_abstract",
                    "anchor_id": "p_abstract",
                    "page_start": 1,
                }
            ]
        },
        locked_citation_source={"sid": "s1234abcd", "source_path": "paper.en.md"},
    )

    assert len(merged) == 2
    assert merged[0]["support_id"] == "DOC-900"
    assert merged[0]["support_example"] == "[[SUPPORT:DOC-900]]"
    assert merged[0]["heading_path"] == "Paper / Abstract"
    assert "tenfold lower" in merged[0]["locate_anchor"]
    assert merged[0]["sid"] == "s1234abcd"
    assert merged[0]["cite_policy"] == "locate_only"
    assert merged[0]["evidence_selection_reason"] == "citation_plan_support_bridge"


def test_shared_primary_evidence_prefers_query_aligned_plan_bridge() -> None:
    exact = (
        "This technique operates at tenfold lower incident illumination power per diffraction "
        "limited spot, significantly reducing photodamage."
    )
    primary = finalize_runtime._pick_shared_primary_evidence(
        paper_guide_contracts_seed={
            "primary_evidence": {
                "source_path": "paper.en.md",
                "heading_path": "Results / Resolution",
                "snippet": "The closed pinhole iPSF has a FWHM of 122 nm.",
                "selection_reason": "answer_hit_top",
            }
        },
        evidence_cards=[],
        support_resolution=[
            {
                "source_path": "paper.en.md",
                "heading_path": "Paper / Abstract",
                "locate_anchor": exact,
                "block_id": "blk_abstract",
                "anchor_id": "p_abstract",
                "evidence_selection_reason": "citation_plan_support_bridge",
            }
        ],
        prompt_text="iISM 在活细胞中是否减少光损伤？",
        answer_text="入射照明功率降低约十倍，从而减少光损伤。",
    )

    assert primary["block_id"] == "blk_abstract"
    assert primary["heading_path"] == "Paper / Abstract"
    assert "tenfold lower" in primary["snippet"]


def test_shared_primary_evidence_keeps_compound_citation_plan_passage() -> None:
    compound = (
        "The operation for digital refocusing can be achieved using two steps. "
        "First, the photon trajectory is reconstructed through a ray tracing operation. "
        "Thus, the second step applies a wave propagation of distance -z to bring the "
        "sample back into focus."
    )
    primary = finalize_runtime._pick_shared_primary_evidence(
        paper_guide_contracts_seed={
            "primary_evidence": {
                "source_path": "qclfm.en.md",
                "heading_path": "A. Concept",
                "snippet": compound.split("Thus,")[0].strip(),
                "selection_reason": "answer_citation",
            },
            "citation_plan": {
                "slots": [
                    {
                        "preferred_system": "system_a",
                        "source_path": "qclfm.en.md",
                        "source_name": "qCLFM.pdf",
                        "heading_path": "A. Concept",
                        "evidence_quote": compound,
                        "block_id": "blk_concept",
                        "anchor_id": "p_refocus",
                        "page_start": 2,
                    }
                ]
            },
        },
        evidence_cards=[],
        support_resolution=[],
        prompt_text="qCLFM 数字重聚焦的两个步骤是什么？",
        answer_text="先做光线追迹，再做波传播逆运算。",
    )

    assert primary["selection_reason"] == "prompt_aligned"
    assert primary["block_id"] == "blk_concept"
    assert "ray tracing" in primary["snippet"]
    assert "wave propagation" in primary["snippet"]


def test_shared_primary_evidence_focuses_long_abstract_on_supporting_sentences() -> None:
    abstract = (
        "Single-pixel cameras measure correlations between a scene and a set of patterns. "
        "These systems often have low frame rates because they require many measurements. "
        "Several compressive sensing techniques mitigate this limitation by undersampling. "
        "Here, we instead exploit the spatiotemporal redundancy of dynamic scenes. "
        "In our system, a high-resolution foveal region tracks motion within the scene, yet "
        "unlike a simple zoom, every frame delivers new spatial information from across the "
        "entire field of view. This strategy records quickly changing features while "
        "accumulating detail of slower regions over several consecutive frames. "
        "The architecture spatially varies resolution and exposure time."
    )

    primary = finalize_runtime._pick_shared_primary_evidence(
        paper_guide_contracts_seed={},
        evidence_cards=[],
        support_resolution=[
            {
                "source_path": "foveated.en.md",
                "heading_path": "Paper / Abstract",
                "locate_anchor": abstract,
                "block_id": "blk_abstract",
                "anchor_id": "p_abstract",
                "evidence_selection_reason": "citation_plan_support_bridge",
            }
        ],
        prompt_text="动态超采样为什么不是普通 zoom？",
        answer_text="中央凹区域跟踪运动，同时每帧覆盖整个视场，并在连续帧中积累细节。",
    )

    assert primary["snippet"].startswith("In our system")
    assert "entire field of view" in primary["snippet"]
    assert "several consecutive frames" in primary["snippet"]
    assert "measure correlations" not in primary["snippet"]


def test_normalize_supported_sequential_terms_keeps_answer_language():
    answer = (
        "Sequential compressed sensing（SCS）利用自适应反馈。\n\n"
        "这种策略保证信号支撑集（support）的恢复。"
    )
    evidence = (
        "A sequential adaptive compressed sensing procedure for signal support recovery is proposed. "
        "The procedure is based on distilled sensing."
    )

    out = finalize_runtime._normalize_citation_plan_supported_terms(
        answer,
        prompt="Sequential compressed sensing 多利用了什么信息？",
        citation_plan={
            "slots": [
                {
                    "preferred_system": "system_a",
                    "evidence_quote": evidence,
                }
            ]
        },
    )

    assert "Sequential adaptive compressed sensing（顺序自适应压缩感知）" in out
    assert "信号支撑集恢复（signal support recovery）" in out
    assert "distilled sensing / 蒸馏感知" in out


def test_normalize_supported_sequential_terms_completes_already_adaptive_label():
    out = finalize_runtime._normalize_citation_plan_supported_terms(
        (
            "Sequential adaptive compressed sensing（顺序自适应压缩感知）相比一次性随机测量，"
            "会利用前一步结果指导下一步测量 [4]。\n\n"
            "该方法主要保证信号支撑集恢复（signal support recovery）。"
        ),
        prompt="顺序压缩感知相比普通压缩感知多利用了什么信息？",
        citation_plan={
            "slots": [
                {
                    "preferred_system": "system_a",
                    "evidence_quote": (
                        "A sequential adaptive compressed sensing procedure for signal support "
                        "recovery is proposed. The procedure is based on distilled sensing."
                    ),
                }
            ]
        },
    )

    assert "distilled sensing / 蒸馏感知" in out
    assert out.count("distilled sensing") == 1
    assert "[4]" in out


def test_normalize_supported_sequential_terms_rewrites_support_set_paraphrase():
    out = finalize_runtime._normalize_citation_plan_supported_terms(
        (
            "Sequential adaptive compressed sensing（顺序自适应压缩感知）基于 distilled sensing。\n\n"
            "它主要保证恢复的是信号的支持集（support recovery），即非零元素的位置。"
        ),
        prompt="它主要保证恢复什么？",
        citation_plan={
            "slots": [
                {
                    "preferred_system": "system_a",
                    "evidence_quote": (
                        "A sequential adaptive compressed sensing procedure for signal support "
                        "recovery is proposed. The procedure is based on distilled sensing."
                    ),
                }
            ]
        },
    )

    assert "主要保证的是信号支撑集恢复（signal support recovery）" in out
    assert "信号的支持集（support recovery）" not in out


def test_normalize_scinerf_training_term_keeps_exact_source_relationship():
    out = finalize_runtime._normalize_citation_plan_supported_terms(
        "具体来说，SCI 的物理成像过程是这样进入训练的：NeRF 渲染后与压缩观测比较。",
        prompt="SCI 的物理成像过程在哪里进入 SCINeRF 训练？",
        citation_plan={
            "slots": [
                {
                    "preferred_system": "system_a",
                    "evidence_quote": (
                        "We formulate the physical imaging process of SCI as part of the "
                        "training of NeRF."
                    ),
                }
            ]
        },
    )

    assert "SCI 的物理成像过程是这样进入 NeRF 训练的" in out


def test_normalize_supported_iism_live_cell_benefit_adds_missing_power_fact():
    out = finalize_runtime._normalize_citation_plan_supported_terms(
        "iISM 在活细胞中实现约 120 nm 横向分辨率。",
        prompt="iISM 对活细胞成像有什么好处？",
        citation_plan={
            "slots": [
                {
                    "preferred_system": "system_a",
                    "evidence_quote": (
                        "The technique achieves about 120 nm lateral resolution while operating at "
                        "tenfold lower incident illumination power per diffraction limited spot, "
                        "significantly reducing photodamage."
                    ),
                }
            ]
        },
    )

    assert "降低约 10 倍" in out
    assert "减少光损伤" in out


def test_normalize_supported_iism_fact_with_spaced_tenfold_value_is_idempotent():
    answer = "论文的 Abstract 报告约 120 nm 分辨率，照明功率降低约 10 倍，从而减少光损伤。 [1]"
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": "iism.en.md",
                "evidence_quote": (
                    "The method achieves 120 nm lateral resolution at tenfold lower incident "
                    "illumination power, significantly reducing photodamage."
                ),
            }
        ]
    }

    once = finalize_runtime._normalize_citation_plan_supported_terms(
        answer,
        prompt="iISM 在活细胞中有什么好处？",
        citation_plan=plan,
        answer_hits=[{"meta": {"source_path": "iism.en.md"}}],
    )
    twice = finalize_runtime._normalize_citation_plan_supported_terms(
        once,
        prompt="iISM 在活细胞中有什么好处？",
        citation_plan=plan,
        answer_hits=[{"meta": {"source_path": "iism.en.md"}}],
    )

    assert twice == once
    assert once.count("Abstract 报告") == 1
    assert once.endswith("[1]")


def test_exact_source_bound_repairs_cover_scope_architecture_tradeoff_and_iism_cost():
    def normalize(answer: str, prompt: str, evidence: str) -> str:
        return finalize_runtime._normalize_citation_plan_supported_terms(
            answer,
            prompt=prompt,
            citation_plan={
                "intent": "scope_boundary" if "perovskite" in prompt else "comparison",
                "slots": [
                    {
                        "preferred_system": "system_a",
                        "source_path": "paper.en.md",
                        "candidate_hits": [1],
                        "evidence_quote": evidence,
                    }
                ],
            },
            answer_hits=[{"text": evidence, "meta": {"source_path": "paper.en.md"}}],
        )

    perovskite = normalize(
        "这篇论文与单像素成像主线关系不大。\n\n它报告了低阈值。",
        "这篇 perovskite laser 和我的单像素成像主线关系大吗？",
        "We demonstrate electrically driven lasing from a dual-cavity perovskite device.",
    )
    assert "dual-cavity perovskite" in perovskite
    assert "而不是单像素成像方法 [1]" in perovskite

    cassi = normalize(
        "## 1. 起点\n\n光谱压缩成像采用双色散器架构 [1]。",
        "SCI 如何从光谱成像走到 3D？",
        "Two dispersive elements are arranged in opposition around a binary-valued aperture.",
    )
    assert "两个相向布置的色散元件" in cassi
    assert "编码孔径快照光谱成像" in cassi
    assert "binary-valued aperture） [1]" in cassi

    cassi_uncited = normalize(
        "CASSI already has two dispersive elements around a binary-valued aperture, "
        "but the model omitted its evidence marker.",
        "What is the CASSI dual-disperser architecture?",
        "Two dispersive elements are arranged in opposition around a binary-valued aperture.",
    )
    assert cassi_uncited.startswith(
        "The verifiable CASSI hardware starts with two dispersive elements arranged "
        "in opposition around a binary-valued aperture [1]."
    )

    s2ism = normalize(
        "传统 ISM 缓解了分辨率与 SNR 的权衡，但厚样本仍会失败 [1]。",
        "s2ISM 打破了什么三方权衡？",
        "ISM overcomes the trade-off between spatial resolution and signal-to-noise ratio, "
        "but does not provide optical sectioning and fails with thick samples unless detector size is limited.",
    )
    assert "空间分辨率与信噪比（SNR）" in s2ism

    iism = normalize(
        "iISM 达到 120 nm [1]。\n\n"
        "这 120 nm 是通过牺牲光照强度换来的 [1]。另一个 122 nm 结果也值得讨论。",
        "iISM 在活细胞中有什么好处，120 nm 是什么代价？",
        "Interferometric detection achieves 120 nm lateral resolution at tenfold lower incident "
        "illumination power per diffraction-limited spot, significantly reducing photodamage.",
    )
    assert "并不是以更高照明功率为代价" in iism
    assert "122 nm" not in iism


def test_exact_source_bound_focuses_iism_live_cell_cost_on_abstract_bundle() -> None:
    evidence = (
        "This next-generation technique combines interferometric detection with image scanning "
        "microscopy to achieve about 120 nm lateral resolution while operating at tenfold lower "
        "incident illumination power per diffraction limited spot, significantly reducing "
        "photodamage while enhancing signal-to-noise and contrast."
    )
    normalized = finalize_runtime._normalize_citation_plan_supported_terms(
        "iISM 达到 120 nm [1]。\n\n另一个 FWHM 122 nm 的结果 [2]。",
        prompt="iISM 在活细胞里同时改善了什么？120 nm 分辨率是用什么代价换来的？",
        citation_plan={
            "slots": [
                {
                    "preferred_system": "system_a",
                    "source_path": "iism.en.md",
                    "candidate_hits": [2],
                    "evidence_quote": evidence,
                }
            ]
        },
        answer_hits=[
            {"text": "Distractor.", "meta": {"source_path": "other.en.md"}},
            {"text": evidence, "meta": {"source_path": "iism.en.md"}},
        ],
    )

    assert "120 nm" in normalized
    assert "降低约 10 倍" in normalized
    assert "photodamage" in normalized
    assert "122 nm" not in normalized
    assert "[2]" in normalized
    assert "[1]" not in normalized


def test_exact_source_bound_stabilizes_iism_depth_phase_roles() -> None:
    relation = (
        "In a confocal geometry, the interference occurs between two quasi-spherical waves "
        "and the relative phase between reflected and scattered electric fields is:"
    )
    variables = (
        "with n the refractive index of the medium, z the axial position of the scatterer "
        "relative to the interface, lambda the illumination wavelength, and phi_Gouy the "
        "Gouy phase."
    )
    out = finalize_runtime._normalize_citation_plan_supported_terms(
        "相位随深度变化。",
        prompt="iISM 的相位为什么携带深度？z、n、λ 和 Gouy phase 分别是什么？",
        citation_plan={
            "slots": [
                {
                    "preferred_system": "system_a",
                    "source_path": "iism.en.md",
                    "candidate_hits": [1],
                    "evidence_quote": relation,
                },
                {
                    "preferred_system": "system_a",
                    "source_path": "iism.en.md",
                    "candidate_hits": [2],
                    "evidence_quote": variables,
                },
            ]
        },
        answer_hits=[
            {"text": relation, "meta": {"source_path": "iism.en.md"}},
            {"text": variables, "meta": {"source_path": "iism.en.md"}},
        ],
    )

    assert all(term in out for term in ("iISM", "反射光", "散射光", "轴向位置", "折射率", "Gouy"))
    assert "[1]" in out
    assert "[2]" in out


def test_source_term_normalization_preserves_spi_image_plane_and_dmd_bottleneck() -> None:
    evidence = (
        "For the latter, the DMD is located in an image plane of the object after a lens. "
        "Nonetheless, it is the modulation rate of the DMD that is the bottleneck in the "
        "acquisition time of a single-pixel camera."
    )
    out = finalize_runtime._normalize_citation_plan_supported_terms(
        "DMD 位于物像平面，采集瓶颈是 DMD 的调制速率 [1]。",
        prompt="结构照明和结构探测有什么差别？速度瓶颈在哪里？",
        citation_plan={
            "slots": [
                {
                    "preferred_system": "system_a",
                    "source_path": "spi.en.md",
                    "candidate_hits": [1],
                    "evidence_quote": evidence,
                }
            ]
        },
        answer_hits=[{"text": evidence, "meta": {"source_path": "spi.en.md"}}],
    )

    assert "像面（image plane）" in out
    assert "DMD 调制速率（modulation rate of the DMD）" in out


def test_exact_source_bound_stabilizes_spi_configuration_and_bottleneck() -> None:
    evidence = (
        "The DMD can project patterns of light onto a scene, also termed structured "
        "illumination, or structure detected image intensities, called structured detection. "
        "For the latter, the DMD is located in an image plane of the object after a lens. "
        "It is the modulation rate of the DMD that is the bottleneck in acquisition time."
    )
    out = finalize_runtime._normalize_citation_plan_supported_terms(
        "两种方式使用 DMD。",
        prompt="structured illumination 和 structured detection 有何区别，速度瓶颈是什么？",
        citation_plan={
            "slots": [
                {
                    "preferred_system": "system_a",
                    "source_path": "spi.en.md",
                    "candidate_hits": [1],
                    "evidence_quote": evidence,
                }
            ]
        },
        answer_hits=[{"text": evidence, "meta": {"source_path": "spi.en.md"}}],
    )

    assert "结构化照明" in out
    assert "结构化探测" in out
    assert "投影（project patterns）" in out
    assert "像面（image plane）" in out
    assert "DMD 调制速率（modulation rate of the DMD）" in out


def test_exact_source_bound_builds_complete_spad_geiger_answer() -> None:
    evidence = (
        "Single photon avalanche diode (SPAD) is a p-n junction that operates in Geiger mode. "
        "The device operates with a bias voltage significantly higher than its reverse bias "
        "breakdown voltage. When the SPAD operates in Geiger mode, excessive induced current "
        "will damage the device's performance, so it must be supported by the quenching circuit. "
        "The circuit detects avalanche current and quench the current by applying an extra "
        "reverse bias."
    )
    normalized = finalize_runtime._normalize_citation_plan_supported_terms(
        "SPAD 工作在盖革模式 [1]。",
        prompt="SPAD 为什么要工作在 Geiger 模式？雪崩之后为什么还需要淬灭电路？",
        citation_plan={
            "slots": [
                {
                    "preferred_system": "system_a",
                    "source_path": "spad.en.md",
                    "candidate_hits": [1],
                    "evidence_quote": evidence,
                }
            ]
        },
        answer_hits=[{"text": evidence, "meta": {"source_path": "spad.en.md"}}],
    )

    assert "Geiger mode" in normalized
    assert "breakdown voltage" in normalized
    assert "quenching circuit（淬灭电路）" in normalized
    assert "额外反向偏置" in normalized
    assert normalized.count("[1]") == 2


def test_exact_source_bound_builds_stable_three_paper_sci_lineage() -> None:
    cassi = "Two dispersive elements are arranged in opposition around a binary-valued aperture."
    scinerf = (
        "We formulate the physical imaging process of SCI as part of the training of NeRF "
        "to recover an underlying 3D scene representation from a single temporal compressed image."
    )
    scigs = (
        "SCIGS is a variant of 3DGS. It reconstructs a dynamic 3D explicit scene from a "
        "single compressed image using a primitive-level transformation network."
    )
    normalized = finalize_runtime._normalize_citation_plan_supported_terms(
        "模型给出了一段混合且不稳定的技术史。",
        prompt="SCI 或压缩快照成像这条线，是怎么从光谱成像走到 3D 场景重建的？",
        citation_plan={
            "intent": "origin_lookup",
            "slots": [
                {
                    "preferred_system": "system_b",
                    "sid": "s_scinerf",
                    "candidate_refs": [50],
                    "evidence_quote": (
                        "video Snapshot Compressive Imaging (SCI) system has emerged to address these limitations."
                    ),
                },
                {
                    "preferred_system": "system_a",
                    "source_path": "scinerf.en.md",
                    "candidate_hits": [1],
                    "evidence_quote": scinerf,
                },
                {
                    "preferred_system": "system_a",
                    "source_path": "scigs.en.md",
                    "candidate_hits": [2],
                    "evidence_quote": scigs,
                },
                {
                    "preferred_system": "system_a",
                    "source_path": "cassi.en.md",
                    "candidate_hits": [3],
                    "evidence_quote": cassi,
                },
            ],
        },
        answer_hits=[
            {"text": scinerf, "meta": {"source_path": "scinerf.en.md"}},
            {"text": scigs, "meta": {"source_path": "scigs.en.md"}},
            {"text": cassi, "meta": {"source_path": "cassi.en.md"}},
        ],
    )

    assert "CASSI 用两个相向布置的色散元件" in normalized
    assert "SCI 的物理成像过程直接纳入 NeRF 训练" in normalized
    assert "SCIGS 进一步把这条路线换成显式 3DGS" in normalized
    assert "[[CITE:s_scinerf:50]]" in normalized
    assert "混合且不稳定" not in normalized


def test_exact_source_bound_completes_beginner_spi_roadmap() -> None:
    prospects = (
        "Their pioneering work has laid the foundations for recovering images from a single-pixel camera "
        "when the number of measurements is fewer than the total number of unknown pixels in the image, "
        "when the properties of the image were sensed compressively, also known as under-sampling or sub-sampling."
    )
    hsi_fsi = (
        "HSI uses Hadamard basis patterns for illumination while FSI uses Fourier basis patterns. "
        "We compare them in terms of principles, imaging efficiency, and noise robustness."
    )
    dl_review = (
        "However, the limited image quality and lengthy computational times for iterative reconstruction still "
        "hinder practical application. Deep learning attracts attention for reconstruction quality and fast "
        "reconstruction speed."
    )
    normalized = finalize_runtime._normalize_citation_plan_supported_terms(
        "前两篇已有证据 [1] [2]，第三篇只有标题。",
        prompt="我刚开始看单像素成像，想先建立主线，应该先读哪几篇？每篇主要看什么？",
        citation_plan={
            "intent": "answer_grounding",
            "slots": [
                {
                    "preferred_system": "system_a",
                    "source_path": "prospects.en.md",
                    "candidate_hits": [1],
                    "evidence_quote": prospects,
                },
                {
                    "preferred_system": "system_a",
                    "source_path": "dl-review.en.md",
                    "candidate_hits": [2],
                    "evidence_quote": dl_review,
                },
                {
                    "preferred_system": "system_a",
                    "source_path": "hsi-fsi.en.md",
                    "candidate_hits": [3],
                    "evidence_quote": hsi_fsi,
                },
            ],
        },
        answer_hits=[
            {"text": prospects, "meta": {"source_path": "prospects.en.md"}},
            {"text": dl_review, "meta": {"source_path": "dl-review.en.md"}},
            {"text": hsi_fsi, "meta": {"source_path": "hsi-fsi.en.md"}},
        ],
    )

    assert "领域框架 → 采样方法选择 → 学习型重建" in normalized
    assert "Principles and prospects for single-pixel imaging" in normalized
    assert "Hadamard single-pixel imaging versus Fourier single-pixel imaging" in normalized
    assert "Advances and Challenges of Single-Pixel Imaging Based on Deep Learning" in normalized
    assert normalized.count("[1]") == 1
    assert normalized.count("[2]") == 1
    assert normalized.count("[3]") == 1
    assert "第三篇只有标题" not in normalized


def test_exact_source_bound_stabilizes_hadamard_fourier_choice() -> None:
    evidence = (
        "Under different sampling ratios, the curves of PSNR, SSIM, and RMSE show that "
        "the convergence of HSI is lower than that of FSI in Fourier space."
    )
    out = finalize_runtime._normalize_citation_plan_supported_terms(
        "追求速度就选 Hadamard [4]。",
        prompt="我做单像素实验，Hadamard 和 Fourier 到底该怎么选？",
        citation_plan={"slots": [{
            "preferred_system": "system_a", "candidate_hits": [1],
            "source_path": "hsi-fsi.en.md", "evidence_quote": evidence,
        }]},
        answer_hits=[{"text": evidence, "meta": {"source_path": "hsi-fsi.en.md"}}],
    )

    assert "不能脱离采样率" in out
    assert "PSNR、SSIM" in out
    assert "测量预算" in out
    assert "[4]" not in out
    assert out.count("[1]") == 2


def test_exact_source_bound_stabilizes_dl_spi_benefits_and_risks() -> None:
    benefit = "Deep learning attracts attention due to exceptional reconstruction quality and fast reconstruction speed."
    risk = "Data-driven strategies face prolonged training duration and limited generalization in diverse imaging scenes."
    out = finalize_runtime._normalize_citation_plan_supported_terms(
        "深度学习速度快 [1]，但也有坑。",
        prompt="深度学习给单像素成像带来的好处和坑分别是什么？",
        citation_plan={"slots": [
            {"preferred_system": "system_a", "candidate_hits": [1], "source_path": "review.en.md", "evidence_quote": benefit},
            {"preferred_system": "system_a", "candidate_hits": [1], "source_path": "review.en.md", "evidence_quote": risk},
        ]},
        answer_hits=[{"text": f"{benefit} {risk}", "meta": {"source_path": "review.en.md"}}],
    )

    assert "重建质量" in out and "重建速度" in out
    assert "数据驱动策略" in out and "泛化能力有限" in out
    assert out.count("[1]") == 2


def test_exact_source_bound_keeps_distinct_same_paper_benefit_and_risk_hits() -> None:
    benefit = "Deep learning attracts attention due to exceptional reconstruction quality and fast reconstruction speed."
    risk = "Data-driven strategies face prolonged training duration and limited generalization in diverse imaging scenes."
    hits = [
        {"text": risk, "meta": {"source_path": "review.en.md", "heading_path": "4. Strategy and Advantages"}},
        {"text": benefit, "meta": {"source_path": "review.en.md", "heading_path": "Abstract"}},
    ]
    out = finalize_runtime._normalize_citation_plan_supported_terms(
        "深度学习速度快 [2]，但也有坑 [1]。",
        prompt="深度学习给单像素成像带来的好处和坑分别是什么？",
        citation_plan={"slots": [
            {
                "preferred_system": "system_a", "candidate_hits": [2],
                "source_path": "review.en.md", "heading_path": "Abstract",
                "evidence_quote": benefit,
            },
            {
                "preferred_system": "system_a", "candidate_hits": [1],
                "source_path": "review.en.md", "heading_path": "4. Strategy and Advantages",
                "evidence_quote": risk,
            },
        ]},
        answer_hits=hits,
    )

    benefit_paragraph = next(part for part in out.split("\n\n") if "重建质量" in part)
    risk_paragraph = next(part for part in out.split("\n\n") if "泛化能力有限" in part)
    assert benefit_paragraph.endswith("[2]。")
    assert "[1]" not in benefit_paragraph
    assert risk_paragraph.count("[1]") == 1
    assert "泛化能力有限） [1]；" in risk_paragraph
    assert "[2]" not in risk_paragraph

    rebound = finalize_runtime._bind_planned_source_citations(
        out,
        citation_plan={"budget": {"system_a": 2}, "slots": [
            {
                "preferred_system": "system_a", "candidate_hits": [2],
                "source_path": "review.en.md", "heading_path": "Abstract",
                "evidence_quote": benefit,
            },
            {
                "preferred_system": "system_a", "candidate_hits": [1],
                "source_path": "review.en.md", "heading_path": "4. Strategy and Advantages",
                "evidence_quote": risk,
            },
        ]},
        answer_hits=hits,
    )
    assert rebound == out


def test_exact_source_bound_uses_richer_dl_spi_challenge_evidence_in_english() -> None:
    benefit = (
        "Deep learning attracts attention due to exceptional reconstruction quality "
        "and fast reconstruction speed."
    )
    risk = (
        "The inherent limitations include reliance on extensive datasets, limited "
        "interpretability, susceptibility to overfitting, and limited generalization."
    )
    hits = [
        {"text": benefit, "meta": {"source_path": "review.en.md"}},
        {"text": risk, "meta": {"source_path": "review.en.md"}},
    ]
    out = finalize_runtime._normalize_citation_plan_supported_terms(
        "Deep learning is useful, but deployment needs care.",
        prompt=(
            "What practical improvements does deep learning bring to single-pixel imaging, "
            "and what limitations should I keep in mind before using it?"
        ),
        citation_plan={
            "slots": [
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [1],
                    "source_path": "review.en.md",
                    "evidence_quote": benefit,
                },
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [2],
                    "source_path": "review.en.md",
                    "evidence_quote": risk,
                },
            ]
        },
        answer_hits=hits,
    )

    assert "reconstruction quality" in out
    assert "training data" in out
    assert "limited interpretability" in out
    assert "overfitting" in out
    assert "limited generalization" in out
    assert out.count("[1]") == 1
    assert out.count("[2]") == 1


def test_exact_source_bound_stabilizes_scinerf_forward_equation() -> None:
    evidence = (
        "$$\\mathbf{Y}=\\sum_i\\mathbf{X}_i\\odot\\mathbf{M}_i+\\mathbf{Z}.$$ "
        "Y is the captured compressed image, Xi is a virtual image, odot denotes "
        "element-wise multiplication, and Z is the measurement noise. Given NeRF and "
        "camera poses, we render Xi to synthesize the compressed image Y, which is "
        "differentiable with respect to NeRF and the poses."
    )
    out = finalize_runtime._normalize_citation_plan_supported_terms(
        "SCINeRF 将多帧压缩成一个测量。",
        prompt=(
            "SCINeRF 的 SCI 前向成像公式到底表达了什么？请解释二值掩模、测量噪声"
            "以及它为什么能进入 NeRF 联合优化。"
        ),
        citation_plan={
            "slots": [
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [1],
                    "source_path": "scinerf.en.md",
                    "evidence_quote": evidence,
                }
            ]
        },
        answer_hits=[{"text": evidence, "meta": {"source_path": "scinerf.en.md"}}],
    )

    assert "\\mathbf{Y}=\\sum" in out
    assert "二值掩模" in out
    assert "逐元素" in out
    assert "测量噪声" in out
    assert "可微" in out
    assert out.count("[1]") == 3


def test_scinerf_formula_answer_is_not_prefixed_by_training_summary() -> None:
    evidence = (
        "The physical imaging process of SCI is part of the training of NeRF. "
        "Y is the captured compressed image, Xi is a virtual image, odot denotes "
        "element-wise multiplication, and Z is the measurement noise. Given NeRF and "
        "camera poses, the synthesized image is differentiable with respect to NeRF and the poses."
    )

    out = finalize_runtime._normalize_citation_plan_supported_terms(
        "SCINeRF compresses virtual frames into one measurement.",
        prompt="What does the SCINeRF forward model equation mean?",
        citation_plan={
            "slots": [
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [1],
                    "source_path": "scinerf.en.md",
                    "evidence_quote": evidence,
                }
            ]
        },
        answer_hits=[{"text": evidence, "meta": {"source_path": "scinerf.en.md"}}],
    )

    assert out.startswith("SCINeRF uses the SCI forward model")
    assert "part of the training of NeRF" not in out.split("\n\n", 1)[0]


def test_fdm_exact_evidence_restores_acquisition_speed_term() -> None:
    evidence = (
        "Frequency-division-multiplexed single-pixel imaging implements frequency-division "
        "methods to parallelize the single-pixel imaging process. The technique enables a "
        "trade-off between signal-to-noise ratio and acquisition speed without altering "
        "detector integration time."
    )

    out = finalize_runtime._normalize_citation_plan_supported_terms(
        "Frequency-division multiplexing parallelizes the imaging process, so it is faster [1].",
        prompt="Why is frequency-division-multiplexed single-pixel imaging faster?",
        citation_plan={
            "slots": [
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [1],
                    "source_path": "Frequency-division-multiplexed single-pixel imaging.en.md",
                    "evidence_quote": evidence,
                }
            ]
        },
        answer_hits=[
            {
                "text": evidence,
                "meta": {
                    "source_path": "Frequency-division-multiplexed single-pixel imaging.en.md"
                },
            }
        ],
    )

    assert "acquisition speed" in out
    assert "unchanged detector integration time" in out


def test_evidence_complete_override_skips_only_full_scinerf_replacement() -> None:
    evidence = (
        "$$\\mathbf{Y}=\\sum_i\\mathbf{X}_i\\odot\\mathbf{M}_i+\\mathbf{Z}.$$ "
        "Y is the captured compressed image, Xi is a virtual image, odot denotes "
        "element-wise multiplication, and Z is the measurement noise. Given NeRF and "
        "camera poses, we render Xi to synthesize the compressed image Y, which is "
        "differentiable with respect to NeRF and the poses."
    )
    out = finalize_runtime._build_evidence_complete_answer_override(
        prompt=(
            "SCINeRF 的 SCI 前向成像公式到底表达了什么？请解释二值掩模、测量噪声"
            "以及它为什么能进入 NeRF 联合优化。"
        ),
        citation_plan={
            "slots": [
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [1],
                    "source_path": "scinerf.en.md",
                    "evidence_quote": evidence,
                }
            ]
        },
        answer_hits=[{"text": evidence, "meta": {"source_path": "scinerf.en.md"}}],
    )

    assert "证据完整回答占位符" not in out
    assert "\\mathbf{Y}=\\sum" in out
    assert "二值掩模" in out
    assert "测量噪声" in out
    assert "可微" in out
    assert out.count("[1]") == 3


def test_evidence_complete_override_keeps_scinerf_physics_inside_training() -> None:
    evidence = (
        "We explore Snapshot Compressive Imaging for recovering the underlying 3D scene "
        "representation from a single temporal compressed image. Specifically, we formulate "
        "the physical imaging process of SCI as part of the training of NeRF, allowing us "
        "to capture complex scene structures."
    )

    out = finalize_runtime._build_evidence_complete_answer_override(
        prompt="SCINeRF 不是先解码视频再跑 NeRF 吗？SCI 的物理成像过程在哪里进入训练？",
        citation_plan={
            "slots": [
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [1],
                    "source_path": "scinerf.en.md",
                    "heading_path": "Abstract",
                    "evidence_quote": evidence,
                }
            ]
        },
        answer_hits=[{"text": evidence, "meta": {"source_path": "scinerf.en.md"}}],
    )

    assert "不是“先把压缩图像解码成视频" in out
    assert "physical imaging process of SCI" in out
    assert "NeRF 训练（training of NeRF）" in out
    assert out.count("[1]") == 2


def test_evidence_complete_override_keeps_partial_repairs_on_model_path() -> None:
    evidence = "The source gives one useful but incomplete detail."

    assert (
        finalize_runtime._build_evidence_complete_answer_override(
            prompt="Explain the method and discuss its limitations.",
            citation_plan={
                "slots": [
                    {
                        "preferred_system": "system_a",
                        "candidate_hits": [1],
                        "source_path": "paper.en.md",
                        "evidence_quote": evidence,
                    }
                ]
            },
            answer_hits=[{"text": evidence, "meta": {"source_path": "paper.en.md"}}],
        )
        == ""
    )


def test_evidence_complete_override_explains_sph_temporal_phase_stepping() -> None:
    evidence = (
        "Two major factors limit the throughput of SPH in current practice: the phase "
        "stepping inherent in holography requires a few patterns for each order. Instead "
        "of actively performing phase shifting, a beat frequency is introduced between "
        "the signal beam and the reference beam, thereby realizing phase stepping naturally "
        "in time by exploiting the framework of heterodyne holography."
    )

    out = finalize_runtime._build_evidence_complete_answer_override(
        prompt="这篇单像素压缩全息怎么提高吞吐量？为什么不再主动做相移？",
        citation_plan={
            "slots": [
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [1],
                    "source_path": "sph.en.md",
                    "heading_path": "Introduction",
                    "evidence_quote": evidence,
                }
            ]
        },
        answer_hits=[{"text": evidence, "meta": {"source_path": "sph.en.md"}}],
    )

    assert "beat frequency（拍频）" in out
    assert "heterodyne holography（外差全息）" in out
    assert "phase stepping naturally in time" in out
    assert out.count("[1]") == 3


def test_evidence_complete_override_keeps_spi_prospects_conditions_and_examples() -> None:
    evidence = (
        "As the approach suits a wide variety of detector technologies, images can be "
        "collected at wavelengths outside the reach of FPA technology or at high frame "
        "rates or in three dimensions. Promising applications include the visualization "
        "of hazardous gas leaks and 3D situation awareness for autonomous vehicles."
    )

    out = finalize_runtime._build_evidence_complete_answer_override(
        prompt="什么场景真的值得用单像素相机，而不是普通面阵相机？有哪些应用？",
        citation_plan={
            "slots": [
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [1],
                    "source_path": "prospects.en.md",
                    "heading_path": "Abstract",
                    "evidence_quote": evidence,
                }
            ]
        },
        answer_hits=[{"text": evidence, "meta": {"source_path": "prospects.en.md"}}],
    )

    assert "wavelengths（波长/波段）" in out
    assert "high frame rates（高帧率）" in out
    assert "three dimensions（三维/3D）" in out
    assert "hazardous gas leaks（危险气体泄漏）" in out
    assert "autonomous vehicles（自动驾驶车辆）" in out
    assert out.count("[1]") == 2


def test_exact_source_bound_stabilizes_fdm_vs_3d_parallelism() -> None:
    fdm = (
        "Each SLM pixel is modulated on p frequencies simultaneously. The system uses "
        "phase-sensitive detection, and the signal is demodulated by p lock-in amplifiers."
    )
    video = (
        "Photometric stereo senses reflected light with four spatially-separated detectors "
        "and reconstructs continuous real-time 3D video at 8 frames per second."
    )
    hits = [
        {"text": fdm, "meta": {"source_path": "fdm.en.md"}},
        {"text": video, "meta": {"source_path": "3d-video.en.md"}},
    ]
    out = finalize_runtime._normalize_citation_plan_supported_terms(
        "Both systems parallelize acquisition.",
        prompt=(
            "Both frequency-division-multiplexed single-pixel imaging and 3D single-pixel "
            "video claim speedups. What does each method parallelize, and why does the 3D "
            "method need multiple detectors?"
        ),
        citation_plan={
            "slots": [
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [1],
                    "source_path": "fdm.en.md",
                    "evidence_quote": fdm,
                },
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [2],
                    "source_path": "3d-video.en.md",
                    "evidence_quote": video,
                },
            ]
        },
        answer_hits=hits,
    )

    assert "phase-sensitive detection" in out
    assert "photometric stereo" in out
    assert "four spatially separated detectors" in out
    assert "8 frames per second" in out
    assert out.count("[1]") == 2
    assert out.count("[2]") == 2


def test_exact_source_bound_stabilizes_sequential_adaptive_scope() -> None:
    evidence = (
        "A sequential adaptive compressed sensing procedure for signal support recovery is proposed. "
        "The procedure is based on distilled sensing and uses sparse sensing matrices for sketching observations."
    )
    out = finalize_runtime._normalize_citation_plan_supported_terms(
        "它使用 distilled sensing [1]。",
        prompt="Sequential compressed sensing 相比一次性随机测量多利用了什么信息？它主要保证恢复什么？",
        citation_plan={"slots": [{
            "preferred_system": "system_a", "candidate_hits": [1],
            "source_path": "seq.en.md", "evidence_quote": evidence,
        }]},
        answer_hits=[{"text": evidence, "meta": {"source_path": "seq.en.md"}}],
    )

    assert "sequential adaptive（顺序自适应）" in out
    assert "signal support recovery（信号支撑集恢复）" in out
    assert "任意图像" in out
    assert out.count("[1]") == 2


def test_exact_source_bound_stabilizes_three_method_microscopy_map() -> None:
    structured = "Structured detection provides super-resolution, high signal-to-noise ratio, and enhanced optical sectioning."
    iism = "Interferometric detection reaches 120 nm at tenfold lower illumination power, reducing photodamage."
    light_field = (
        "Light-field microscopy gains volumetric information in a single shot by "
        "simultaneously capturing both position and angular information."
    )
    hits = [
        {"text": light_field, "meta": {"source_path": "qclfm.en.md"}},
        {"text": iism, "meta": {"source_path": "iism.en.md"}},
        {"text": structured, "meta": {"source_path": "s2ism.en.md"}},
    ]
    out = finalize_runtime._normalize_citation_plan_supported_terms(
        "模型输出被截断。",
        prompt="显微成像这些 structured detection、interferometric、light-field 方法分别是在解决什么麻烦？",
        citation_plan={"intent": "comparison", "slots": [
            {"preferred_system": "system_a", "candidate_hits": [1], "source_path": "qclfm.en.md", "evidence_quote": light_field},
            {"preferred_system": "system_a", "candidate_hits": [2], "source_path": "iism.en.md", "evidence_quote": iism},
            {"preferred_system": "system_a", "candidate_hits": [3], "source_path": "s2ism.en.md", "evidence_quote": structured},
        ]},
        answer_hits=hits,
    )

    assert "Structured detection" in out and "SNR" in out
    assert "Interferometric detection" in out and "120 nm" in out
    assert "Light-field" in out and "refocus" in out
    assert out.count("[1]") == 3
    assert out.count("[2]") == 2
    assert out.count("[3]") == 2
    assert "模型输出被截断" not in out


def test_late_target_hits_rebuilds_basis_foveated_pair(tmp_path: Path) -> None:
    foveated = tmp_path / "SciAdv-2017-Adaptive foveated single-pixel imaging with dynamic supersampling.en.md"
    foveated.write_text(
        "<!-- kb_page: 1 -->\n\n## Abstract\n\n"
        "A high-resolution foveal region tracks motion, yet every frame delivers new spatial information "
        "from across the entire field of view while slower regions accumulate detail over consecutive frames.\n\n"
        "<!-- kb_page: 3 -->\n\n## Method\n\nGeneric supersampling body text.\n",
        encoding="utf-8",
    )
    basis = tmp_path / "Hadamard single-pixel imaging versus Fourier single-pixel imaging.en.md"
    basis.write_text(
        "<!-- kb_page: 3 -->\n\n## Introduction\n\n"
        "HSI uses Hadamard basis patterns for illumination while FSI uses Fourier basis patterns. "
        "The paper compares principles, imaging efficiency, and noise robustness.\n",
        encoding="utf-8",
    )
    hits = [
        {"text": "Generic supersampling body text.", "meta": {"source_path": str(foveated)}},
        {"text": "HSI and FSI comparison.", "meta": {"source_path": str(basis)}},
    ]
    rebuilt = finalize_runtime._citation_plan_with_late_target_hits(
        {"budget": {"system_a": 2, "system_b": 0}, "slots": []},
        answer_hits=hits,
        prompt="Hadamard/Fourier 和 foveated dynamic supersampling 是同一层面吗？分别决定什么？",
    )
    slots = list(rebuilt.get("slots") or [])

    assert len(slots) == 2
    assert any("Hadamard basis patterns" in str(slot.get("evidence_quote")) for slot in slots)
    assert any("entire field of view" in str(slot.get("evidence_quote")) for slot in slots)
    assert any(str(slot.get("heading_path") or "").endswith("Abstract") for slot in slots)


def test_late_target_hits_preserves_structured_table_metric_evidence() -> None:
    evidence = (
        "Table 6 | SIDD | Baseline ours | 40.30 PSNR; "
        "NAFNet ours | 40.30 PSNR"
    )
    hits = [
        {
            "text": evidence,
            "meta": {
                "source_path": "simple-baselines.en.md",
                "source_name": "Simple Baselines for Image Restoration",
                "heading_path": "Experiments / Table 6",
                "structured_kind": "table_metric",
                "block_id": "table-6",
                "anchor_id": "table-6-sidd-psnr",
                "anchor_kind": "table",
                "page_start": 8,
            },
        }
    ]

    rebuilt = finalize_runtime._citation_plan_with_late_target_hits(
        {
            "slots": [
                {
                    "preferred_system": "system_a",
                    "source_path": "simple-baselines.en.md",
                    "heading_path": "Introduction",
                    "evidence_quote": "A generic restoration overview.",
                }
            ]
        },
        answer_hits=hits,
        prompt="SIDD 的 PSNR 最高模型是谁？并列请全部列出。",
    )

    slot = rebuilt["slots"][0]
    assert slot["candidate_hits"] == [1]
    assert slot["evidence_quote"] == evidence
    assert slot["heading_path"] == "Experiments / Table 6"
    assert slot["block_id"] == "table-6"
    assert slot["anchor_kind"] == "table"
    assert slot["page_start"] == 8
    assert rebuilt["late_target_hit_refresh"] is True


def test_late_target_hits_separates_dl_benefit_and_risk_passages(tmp_path: Path) -> None:
    source = tmp_path / "LPR-2025-Advances and Challenges of Single-Pixel Imaging Based on Deep Learning.en.md"
    source.write_text(
        "<!-- kb_page: 1 -->\n\n## Abstract\n\n"
        "However, limited image quality and lengthy computational times for iterative reconstruction hinder use. "
        "Single-pixel imaging based on deep learning has exceptional reconstruction quality and fast reconstruction speed.\n\n"
        "<!-- kb_page: 8 -->\n\n## 4. Strategy and Advantages / Data-Driven Strategy\n\n"
        "Data-driven strategies have prolonged training duration and limited generalization in diverse imaging scenes.\n",
        encoding="utf-8",
    )
    hits = [
        {"text": "Data-driven strategies have prolonged training duration and limited generalization.", "meta": {"source_path": str(source), "heading_path": "4. Strategy and Advantages / Data-Driven Strategy"}},
        {"text": "Single-pixel imaging based on deep learning has exceptional reconstruction quality and fast reconstruction speed.", "meta": {"source_path": str(source), "heading_path": "Abstract"}},
    ]
    rebuilt = finalize_runtime._citation_plan_with_late_target_hits(
        {"budget": {"system_a": 2, "system_b": 0}, "slots": []},
        answer_hits=hits,
        prompt="深度学习给单像素成像带来的好处和坑分别是什么？",
    )
    slots = list(rebuilt.get("slots") or [])

    assert len(slots) == 2
    assert {tuple(slot.get("candidate_hits") or []) for slot in slots} == {(1,), (2,)}
    surface = "\n".join(str(slot.get("evidence_quote") or "") for slot in slots)
    assert "reconstruction quality" in surface
    assert "limited generalization" in surface
    risk_slot = next(
        slot for slot in slots if "limited generalization" in str(slot.get("evidence_quote") or "")
    )
    assert "Strategy and Advantages" in str(risk_slot.get("heading_path") or "")


def test_late_target_hits_prefers_rich_dl_challenges_for_english_prompt() -> None:
    benefit = (
        "Deep learning has exceptional reconstruction quality and fast reconstruction speed."
    )
    strategy = (
        "Data-driven strategies have prolonged training duration and limited generalization."
    )
    challenges = (
        "The limitations include reliance on extensive datasets, limited interpretability, "
        "susceptibility to overfitting, and limited generalization during training."
    )
    hits = [
        {
            "text": challenges,
            "meta": {
                "source_path": "review.en.md",
                "heading_path": "6. Challenges and Outlooks",
            },
        },
        {
            "text": strategy,
            "meta": {
                "source_path": "review.en.md",
                "heading_path": "4. Strategy and Advantages / Data-Driven Strategy",
            },
        },
        {
            "text": benefit,
            "meta": {"source_path": "review.en.md", "heading_path": "Abstract"},
        },
    ]

    rebuilt = finalize_runtime._citation_plan_with_late_target_hits(
        {"budget": {"system_a": 2, "system_b": 0}, "slots": []},
        answer_hits=hits,
        prompt=(
            "What practical improvements does deep learning bring, and what limitations "
            "should I keep in mind?"
        ),
    )
    risk_slot = next(
        slot
        for slot in list(rebuilt.get("slots") or [])
        if "limited generalization" in str(slot.get("evidence_quote") or "")
    )

    assert risk_slot["candidate_hits"] == [1]
    assert risk_slot["heading_path"] == "6. Challenges and Outlooks"


def test_normalize_supported_iism_marker_across_private_public_source_paths():
    evidence = (
        "The method achieves 120 nm lateral resolution at tenfold lower incident "
        "illumination power, significantly reducing photodamage."
    )
    private_path = "F:/corpus/db/iism/iism.en.md"
    public_path = "kb-source/0/iism/iism.en.md"
    plan = {
        "budget": {"system_a": 1, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": private_path,
                "heading_path": "Abstract",
                "evidence_quote": evidence,
            }
        ],
    }
    hits = [
        {
            "text": evidence if index == 1 else "Same-paper retrieval passage.",
            "meta": {
                "source_path": public_path,
                "heading_path": "Abstract" if index == 1 else "Results",
            },
        }
        for index in range(1, 5)
    ]

    out = finalize_runtime._normalize_citation_plan_supported_terms(
        "论文的 Abstract 报告约 120 nm 分辨率，照明功率降低约 10 倍，从而减少光损伤 [4]。",
        prompt="iISM 在活细胞中有什么好处？",
        citation_plan=plan,
        answer_hits=hits,
    )
    out = finalize_runtime._collapse_adjacent_duplicate_numeric_citations(out)

    assert out.count("[1]") == 1
    assert "[4]" not in out


def test_normalize_supported_scigs_variant_uses_exact_answer_hit():
    out = finalize_runtime._normalize_citation_plan_supported_terms(
        "SCIGS 的核心新意是从单张压缩图重建动态 3D 场景 [3]。",
        prompt="SCIGS 的核心新意是什么？",
        citation_plan={"slots": []},
        answer_hits=[{"text": "The proposed SCIGS is a variant of 3DGS."}],
    )

    assert "SCIGS 是面向 SCI 的 3DGS 变体" in out
    assert "3DGS 变体 [3]" in out
    assert out.count("[3]") == 2


def test_normalize_supported_scigs_adds_single_image_term_and_drops_unasked_named_comparison():
    out = finalize_runtime._normalize_citation_plan_supported_terms(
        (
            "SCIGS 是面向 SCI 的 3DGS 变体 [1]。\n\n"
            "具体来说，SCIGS 声称：\n- 输入：仅需一张动态场景的压缩图像。\n"
            "- 现有 NeRF 方法（如 SCINeRF）难以处理动态场景。"
        ),
        prompt="SCIGS 的核心新意是什么？",
        citation_plan={
            "budget": {"system_a": 2, "system_b": 0},
            "slots": [
                {
                    "preferred_system": "system_a",
                    "source_path": "scigs.en.md",
                    "evidence_quote": (
                        "SCIGS is a variant of 3DGS and reconstructs dynamic 3D scenes "
                        "from a single compressed image."
                    ),
                }
            ],
        },
        answer_hits=[{"meta": {"source_path": "scigs.en.md"}}],
    )

    assert "单张压缩图像（single compressed image）" in out
    assert "SCINeRF" not in out


def test_scigs_scinerf_comparison_adds_exact_planned_method_fact_at_resolved_source_hit() -> None:
    scinerf_quote = (
        "Specifically, we formulate the physical imaging process of SCI as part "
        "of the training of NeRF, allowing us to capture complex scene structures."
    )
    scinerf_path = "F:/kb/db/scinerf/scinerf.en.md"
    scigs_path = "F:/kb/db/scigs/scigs.en.md"
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": scinerf_path,
                "source_name": "SCINeRF",
                "heading_path": "SCINeRF / Abstract",
                "evidence_quote": scinerf_quote,
                # The plan number predates canonical-hit recovery; source-local
                # evidence must resolve this to the exact Abstract hit at [3].
                "candidate_hits": [2],
            },
            {
                "preferred_system": "system_a",
                "source_path": scigs_path,
                "source_name": "SCIGS",
                "heading_path": "SCIGS / Abstract",
                "evidence_quote": "SCIGS reconstructs explicit dynamic 3D scenes.",
                "candidate_hits": [4],
            },
        ],
    }
    hits = [
        {"text": "SCIGS title.", "meta": {"source_path": scigs_path}},
        {"text": "SCINeRF title.", "meta": {"source_path": scinerf_path}},
        {"text": scinerf_quote, "meta": {"source_path": scinerf_path, "heading_path": "SCINeRF / Abstract"}},
        {"text": "SCIGS reconstructs explicit dynamic 3D scenes.", "meta": {"source_path": scigs_path}},
    ]
    answer = (
        "SCIGS \u4ece\u5355\u5f20\u538b\u7f29\u56fe\u50cf\u91cd\u5efa\u663e\u5f0f\u52a8\u6001 3D \u573a\u666f [4]\u3002\n\n"
        "SCINeRF \u57fa\u4e8e\u9690\u5f0f NeRF \u8868\u793a\uff0cSCIGS \u5219\u57fa\u4e8e\u663e\u5f0f 3DGS \u8868\u793a\u3002"
    )

    once = finalize_runtime._normalize_scigs_scinerf_plan_comparison_claim(
        answer,
        prompt="SCIGS \u60f3\u89e3\u51b3\u4ec0\u4e48\uff1f\u5b83\u548c SCINeRF \u7684\u533a\u522b\u5728\u54ea\u91cc\uff1f",
        citation_plan=plan,
        answer_hits=hits,
    )
    twice = finalize_runtime._normalize_scigs_scinerf_plan_comparison_claim(
        once,
        prompt="SCIGS \u60f3\u89e3\u51b3\u4ec0\u4e48\uff1f\u5b83\u548c SCINeRF \u7684\u533a\u522b\u5728\u54ea\u91cc\uff1f",
        citation_plan=plan,
        answer_hits=hits,
    )

    assert "SCINeRF \u5219\u628a SCI \u7684\u7269\u7406\u6210\u50cf\u8fc7\u7a0b\u4f5c\u4e3a NeRF \u8bad\u7ec3\u7684\u4e00\u90e8\u5206 [3]\u3002" in once
    assert twice == once


def test_scigs_scinerf_comparison_does_not_add_fact_without_exact_plan_evidence() -> None:
    answer = "SCIGS uses explicit 3DGS, whereas SCINeRF uses an implicit NeRF representation."
    out = finalize_runtime._normalize_scigs_scinerf_plan_comparison_claim(
        answer,
        prompt="How does SCIGS differ from SCINeRF?",
        citation_plan={
            "intent": "comparison",
            "slots": [
                {
                    "preferred_system": "system_a",
                    "source_path": "scinerf.en.md",
                    "source_name": "SCINeRF",
                    "evidence_quote": "SCINeRF recovers a neural radiance field from a compressed image.",
                    "candidate_hits": [1],
                },
                {
                    "preferred_system": "system_a",
                    "source_path": "scigs.en.md",
                    "source_name": "SCIGS",
                    "evidence_quote": "SCIGS uses an explicit 3DGS representation.",
                    "candidate_hits": [2],
                },
            ],
        },
        answer_hits=[
            {"text": "SCINeRF evidence.", "meta": {"source_path": "scinerf.en.md"}},
            {"text": "SCIGS evidence.", "meta": {"source_path": "scigs.en.md"}},
        ],
    )

    assert out == answer
    assert "physical imaging process" not in out


def test_scigs_scinerf_comparison_uses_english_sentence_for_english_answer() -> None:
    evidence = (
        "We formulate the physical imaging process of SCI as part of the training of NeRF."
    )
    out = finalize_runtime._normalize_scigs_scinerf_plan_comparison_claim(
        "SCIGS uses explicit 3DGS, whereas SCINeRF uses an implicit NeRF representation.",
        prompt="How does SCIGS differ from SCINeRF?",
        citation_plan={
            "intent": "comparison",
            "slots": [
                {
                    "preferred_system": "system_a",
                    "source_path": "scinerf.en.md",
                    "source_name": "SCINeRF",
                    "evidence_quote": evidence,
                    "candidate_hits": [1],
                },
                {
                    "preferred_system": "system_a",
                    "source_path": "scigs.en.md",
                    "source_name": "SCIGS",
                    "evidence_quote": "SCIGS uses an explicit 3DGS representation.",
                    "candidate_hits": [2],
                },
            ],
        },
        answer_hits=[
            {"text": evidence, "meta": {"source_path": "scinerf.en.md"}},
            {"text": "SCIGS uses an explicit 3DGS representation.", "meta": {"source_path": "scigs.en.md"}},
        ],
    )

    assert "SCINeRF formulates the physical imaging process of SCI as part of the training of NeRF [1]." in out


def test_scigs_scinerf_comparison_does_not_repeat_equivalent_forward_model_fact() -> None:
    evidence = (
        "We formulate the physical imaging process of SCI as part of the training of NeRF."
    )
    answer = (
        "SCIGS uses explicit 3DGS. SCINeRF embeds the SCI forward model directly "
        "in NeRF optimization."
    )
    out = finalize_runtime._normalize_scigs_scinerf_plan_comparison_claim(
        answer,
        prompt="How does SCIGS differ from SCINeRF?",
        citation_plan={
            "slots": [
                {
                    "preferred_system": "system_a",
                    "source_path": "scinerf.en.md",
                    "source_name": "SCINeRF",
                    "evidence_quote": evidence,
                    "candidate_hits": [1],
                },
                {
                    "preferred_system": "system_a",
                    "source_path": "scigs.en.md",
                    "source_name": "SCIGS",
                    "evidence_quote": "SCIGS uses an explicit 3DGS representation.",
                    "candidate_hits": [2],
                },
            ]
        },
        answer_hits=[
            {"text": evidence, "meta": {"source_path": "scinerf.en.md"}},
            {"text": "SCIGS uses explicit 3DGS.", "meta": {"source_path": "scigs.en.md"}},
        ],
    )

    assert out == answer


def test_scigs_scinerf_comparison_never_appends_fact_to_markdown_heading() -> None:
    evidence = (
        "We formulate the physical imaging process of SCI as part of the training of NeRF."
    )
    out = finalize_runtime._normalize_scigs_scinerf_plan_comparison_claim(
        "### SCIGS vs SCINeRF / NeRF",
        prompt="How does SCIGS differ from SCINeRF?",
        citation_plan={
            "slots": [
                {
                    "preferred_system": "system_a",
                    "source_path": "scinerf.en.md",
                    "source_name": "SCINeRF",
                    "evidence_quote": evidence,
                    "candidate_hits": [1],
                },
                {
                    "preferred_system": "system_a",
                    "source_path": "scigs.en.md",
                    "source_name": "SCIGS",
                    "evidence_quote": "SCIGS uses an explicit 3DGS representation.",
                    "candidate_hits": [2],
                },
            ]
        },
        answer_hits=[
            {"text": evidence, "meta": {"source_path": "scinerf.en.md"}},
            {"text": "SCIGS uses explicit 3DGS.", "meta": {"source_path": "scigs.en.md"}},
        ],
    )

    lines = out.splitlines()
    assert lines[0] == "### SCIGS vs SCINeRF / NeRF"
    assert lines[1] == ""
    assert lines[2].startswith("SCINeRF formulates")


def test_scigs_scinerf_heading_context_prevents_repeating_anaphoric_fact() -> None:
    evidence = (
        "We formulate the physical imaging process of SCI as part of the training of NeRF."
    )
    answer = (
        "### SCIGS vs SCINeRF / NeRF\n\n"
        "The latter embeds the SCI forward model into its optimization."
    )
    out = finalize_runtime._normalize_scigs_scinerf_plan_comparison_claim(
        answer,
        prompt="How does SCIGS differ from SCINeRF?",
        citation_plan={
            "slots": [
                {
                    "preferred_system": "system_a",
                    "source_path": "scinerf.en.md",
                    "source_name": "SCINeRF",
                    "evidence_quote": evidence,
                    "candidate_hits": [1],
                },
                {
                    "preferred_system": "system_a",
                    "source_path": "scigs.en.md",
                    "source_name": "SCIGS",
                    "evidence_quote": "SCIGS uses an explicit 3DGS representation.",
                    "candidate_hits": [2],
                },
            ]
        },
        answer_hits=[
            {"text": evidence, "meta": {"source_path": "scinerf.en.md"}},
            {"text": "SCIGS uses explicit 3DGS.", "meta": {"source_path": "scigs.en.md"}},
        ],
    )

    assert out == answer


def test_finalize_scigs_scinerf_comparison_keeps_added_plan_fact_grounded() -> None:
    scinerf_quote = (
        "Specifically, we formulate the physical imaging process of SCI as part "
        "of the training of NeRF, allowing us to capture complex scene structures."
    )
    scinerf_path = "F:/kb/db/scinerf/scinerf.en.md"
    scigs_path = "F:/kb/db/scigs/scigs.en.md"
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": scinerf_path,
                "source_name": "SCINeRF",
                "heading_path": "SCINeRF / Abstract",
                "evidence_quote": scinerf_quote,
                "candidate_hits": [2],
            },
            {
                "preferred_system": "system_a",
                "source_path": scigs_path,
                "source_name": "SCIGS",
                "heading_path": "SCIGS / Abstract",
                "evidence_quote": "SCIGS reconstructs explicit dynamic 3D scenes.",
                "candidate_hits": [4],
            },
        ],
    }
    hits = [
        {"text": "SCIGS title.", "meta": {"source_path": scigs_path}},
        {"text": "SCINeRF title.", "meta": {"source_path": scinerf_path}},
        {
            "text": scinerf_quote,
            "meta": {
                "source_path": scinerf_path,
                "heading_path": "SCINeRF / Abstract",
            },
        },
        {
            "text": "SCIGS reconstructs explicit dynamic 3D scenes.",
            "meta": {"source_path": scigs_path, "heading_path": "SCIGS / Abstract"},
        },
    ]

    out = finalize_runtime._finalize_generation_answer(
        (
            "SCIGS \u4ece\u5355\u5f20\u538b\u7f29\u56fe\u50cf\u91cd\u5efa\u663e\u5f0f\u52a8\u6001 3D \u573a\u666f [10004]\u3002\n\n"
            "SCINeRF \u57fa\u4e8e\u9690\u5f0f NeRF \u8868\u793a\uff0cSCIGS \u5219\u57fa\u4e8e\u663e\u5f0f 3DGS \u8868\u793a\u3002"
        ),
        prompt="SCIGS \u60f3\u89e3\u51b3\u4ec0\u4e48\uff1f\u5b83\u548c SCINeRF \u7684\u533a\u522b\u5728\u54ea\u91cc\uff1f",
        prompt_for_user="SCIGS \u60f3\u89e3\u51b3\u4ec0\u4e48\uff1f\u5b83\u548c SCINeRF \u7684\u533a\u522b\u5728\u54ea\u91cc\uff1f",
        answer_hits=hits,
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
        paper_guide_contracts_seed={"citation_plan": plan},
        apply_paper_guide_answer_postprocess=lambda answer, **_kwargs: (answer, []),
        maybe_append_library_figure_markdown=lambda answer, **_kwargs: answer,
        validate_structured_citations=lambda answer, **_kwargs: (answer, {}),
    )

    assert "SCINeRF \u5219\u628a SCI \u7684\u7269\u7406\u6210\u50cf\u8fc7\u7a0b\u4f5c\u4e3a NeRF \u8bad\u7ec3\u7684\u4e00\u90e8\u5206 [3]\u3002" in out["answer"]
    assert out["answer_quality"]["claim_evidence"]["minimum_ok"] is True
    assert out["answer_quality"]["claim_evidence"]["citation_mismatch_claims"] == 0


def test_normalize_supported_terms_matches_virtual_and_absolute_source_paths():
    plan_path = r"F:\repo\db\paper\paper.en.md"
    hit_path = "kb-source/0/paper/paper.en.md"
    out = finalize_runtime._normalize_citation_plan_supported_terms(
        "动态超采样和普通 zoom 不同：它让每帧都采集整个视场，并融合连续多帧。",
        prompt="动态超采样和普通 zoom 的关键区别是什么？",
        citation_plan={
            "slots": [
                {
                    "preferred_system": "system_a",
                    "source_path": plan_path,
                    "evidence_quote": (
                        "A high-resolution foveal region tracks motion. Unlike a simple zoom, "
                        "every frame delivers new spatial information from across the entire field of view."
                    ),
                }
            ]
        },
        answer_hits=[{"meta": {"source_path": hit_path}}],
    )

    assert "foveal region" in out
    assert out.endswith("[1]")


def test_foveated_grounded_supplement_keeps_direct_answer_first():
    out = finalize_runtime._normalize_citation_plan_supported_terms(
        (
            "不完全对。动态超采样不是简单地只对重点区域多拍，"
            "而是通过跨帧互补采样提升空间分辨率。"
        ),
        prompt="动态超采样是不是只盯着重要区域多拍？",
        citation_plan={
            "slots": [
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [1],
                    "source_path": "foveated.en.md",
                    "heading_path": "Foveated / Abstract",
                    "page_start": 1,
                    "evidence_quote": (
                        "A high-resolution foveal region tracks motion. Unlike a simple "
                        "zoom, every frame delivers new spatial information from across "
                        "the entire field of view while detail accumulates over several "
                        "consecutive frames."
                    ),
                }
            ]
        },
        answer_hits=[
            {
                "meta": {
                    "source_path": "foveated.en.md",
                    "ref_answer_citation_num": 1,
                }
            }
        ],
    )

    paragraphs = out.split("\n\n")
    assert paragraphs[0].startswith("不完全对")
    assert "foveal region" in paragraphs[0]
    assert "整个视场" in paragraphs[0]
    assert "连续多帧" in paragraphs[0]
    assert out.count("foveal region") == 1


def test_foveated_exact_whole_field_clause_gets_occurrence_specific_citation():
    out = finalize_runtime._normalize_citation_plan_supported_terms(
        (
            "普通 zoom 只放大局部；动态超采样的每一帧都从整个视场采集新的空间信息，"
            "并通过连续多帧累积慢变化区域的细节 [3]。"
        ),
        prompt="动态超采样和普通 zoom 有什么区别？",
        citation_plan={
            "budget": {"system_a": 2, "system_b": 0},
            "slots": [
                {
                    "preferred_system": "system_a",
                    "source_path": "foveated.en.md",
                    "evidence_quote": (
                        "A high-resolution foveal region tracks motion. Unlike a simple zoom, every frame "
                        "delivers new spatial information from across the entire field of view while detail "
                        "accumulates over several consecutive frames."
                    ),
                }
            ],
        },
        answer_hits=[
            {"meta": {"source_path": "other.en.md"}},
            {"meta": {"source_path": "other-2.en.md"}},
            {"meta": {"source_path": "kb-source/0/foveated/foveated.en.md"}},
        ],
    )

    assert "foveal region" in out
    assert "连续多帧中为慢变区域累积细节。 [3]" in out
    assert out.count("[3]") == 1


def test_normalize_supported_terms_replaces_missing_table_values_from_plan_rows():
    evidence = (
        "Table 1. Results. Algorithm: ISTA-Net. CS Ratio 50% = 37.43; "
        "CS Ratio 25% = 31.53; Time CPU/GPU = 0.923s/0.039s; "
        "FPS CPU/GPU = 1.08/25.6 "
        "Table 1. Results. Algorithm: ISTA-Net$^+$. CS Ratio 50% = 38.07; "
        "CS Ratio 25% = 32.57; Time CPU/GPU = 1.375s/0.047s; "
        "FPS CPU/GPU = 0.73/21.3"
    )
    out = finalize_runtime._normalize_citation_plan_supported_terms(
        (
            "根据原文表1：\n\n"
            "| 方法 | PSNR (dB) | CPU时间 | GPU时间 | FPS |\n"
            "| --- | --- | --- | --- | --- |\n"
            "| ISTA-Net | 未在检索片段中显示 | — | — | 1.08/25.6 [1] |\n\n"
            "**说明**：当前片段缺少完整数据。"
        ),
        prompt=(
            "《ISTA-Net》表1里 Set11、25% CS ratio 时，ISTA-Net 与 ISTA-Net+ "
            "的 PSNR、CPU/GPU 时间和 FPS 分别是多少？"
        ),
        citation_plan={
            "slots": [
                {
                    "preferred_system": "system_a",
                    "source_path": "ista-net.en.md",
                    "candidate_hits": [1],
                    "evidence_quote": evidence,
                }
            ]
        },
        answer_hits=[{"meta": {"source_path": "ista-net.en.md"}}],
    )

    assert "| ISTA-Net | 31.53 | 0.923s | 0.039s | 1.08/25.6 [1] |" in out
    assert "| ISTA-Net$^+$ | 32.57 | 1.375s | 0.047s | 0.73/21.3 [1] |" in out
    assert "未在检索片段中显示" not in out
    assert "当前片段缺少完整数据" not in out


def test_normalize_supported_terms_restores_complete_degradation_chain() -> None:
    evidence = (
        "The illumination patterns from the projector undergo scattering and non-ideal focus, "
        "introducing blur during the illumination stage. The modulated light pattern is projected "
        "onto the object, where spatial downsampling occurs due to the limited resolution of the patterns. "
        "Mechanical jitters between the object and projection system introduce relative misalignment, "
        "leading to multiplicative fluctuations in the measurement. The reflected light may experience "
        "additional degradation along the detection path due to scattering imperfections, resulting in "
        "further blur. Finally, photon shot noise and electronic noise affect the detection process; photon "
        "shot noise is modeled as a Poisson-distributed function. As the single-pixel detector integrates "
        "the collected light intensities from the entire scene, noise from each photodetector readout can "
        "propagate and spread to the entire image after reconstruction."
    )
    out = finalize_runtime._normalize_citation_plan_supported_terms(
        "真实退化链包括空间下采样。局部读出噪声会传播为全局污染 [1]。",
        prompt="真实退化链有哪些环节？为什么局部读出噪声会变成全局图像污染？",
        citation_plan={
            "slots": [
                {
                    "preferred_system": "system_a",
                    "source_path": "degradation.en.md",
                    "candidate_hits": [1],
                    "evidence_quote": evidence,
                }
            ]
        },
        answer_hits=[{"meta": {"source_path": "degradation.en.md"}}],
    )

    for term in ("非理想聚焦", "空间下采样", "机械抖动", "探测路径", "光子散粒噪声", "电子噪声"):
        assert term in out
    assert "整个场景光强的积分值" in out
    assert "传播到整幅图像" in out


def test_normalize_supported_iism_fact_gets_existing_system_a_citation():
    out = finalize_runtime._normalize_citation_plan_supported_terms(
        (
            "iISM 实现约 120 nm 分辨率 [1]。\n\n"
            "在约 120 nm 分辨率下，入射照明功率降低约 10 倍，从而减少光损伤。"
        ),
        prompt="iISM 在活细胞里同时改善了什么？",
        citation_plan={
            "slots": [
                {
                    "preferred_system": "system_a",
                    "source_path": "paper.en.md",
                    "evidence_quote": (
                        "The method achieves 120 nm lateral resolution at tenfold lower incident "
                        "illumination power, significantly reducing photodamage."
                    ),
                }
            ]
        },
        answer_hits=[{"meta": {"source_path": "paper.en.md"}}],
    )

    assert "减少光损伤。 [1]" in out


def test_normalize_supported_qclfm_adds_cited_two_step_claim() -> None:
    evidence = (
        "The operation for digital refocusing can be achieved using two steps. "
        "First, the trajectory of the photons is reconstructed through a ray tracing operation. "
        "The second step reverses diffraction by applying wave propagation of distance -z."
    )
    out = finalize_runtime._normalize_citation_plan_supported_terms(
        "QCLFM 通过两步数字重聚焦将离焦样品重新对焦。\n\n第一步使用光线追迹。\n\n第二步反演衍射。 [1]",
        prompt="QCLFM 如何做数字重聚焦？",
        citation_plan={
            "slots": [
                {
                    "preferred_system": "system_a",
                    "source_path": "qclfm.en.md",
                    "heading_path": "A. Concept",
                    "evidence_quote": evidence,
                }
            ]
        },
        answer_hits=[{"meta": {"source_path": "qclfm.en.md"}}],
    )

    compound = next(paragraph for paragraph in out.split("\n\n") if "ray tracing" in paragraph)
    assert "wave propagation" in compound
    assert "重聚焦" in compound
    assert compound.endswith("[1]")


def test_normalize_supported_dl_benefit_and_risk_cites_both_claims() -> None:
    out = finalize_runtime._normalize_citation_plan_supported_terms(
        (
            "好处：深度学习带来卓越的重建质量和快速重建速度。\n\n"
            "局限：数据驱动策略训练时间长、泛化能力有限。 [1]"
        ),
        prompt="深度学习给单像素成像带来的好处和坑分别是什么？",
        citation_plan={
            "slots": [
                {
                    "preferred_system": "system_a",
                    "source_path": "dl-spi.en.md",
                    "heading_path": "Abstract",
                    "evidence_quote": (
                        "Deep learning offers exceptional reconstruction quality and fast reconstruction speed."
                    ),
                },
                {
                    "preferred_system": "system_a",
                    "source_path": "dl-spi.en.md",
                    "heading_path": "4. Strategy and Advantages",
                    "evidence_quote": (
                        "Data-driven strategies require prolonged training and have limited generalization."
                    ),
                },
            ]
        },
        answer_hits=[{"meta": {"source_path": "dl-spi.en.md"}}],
    )

    paragraphs = out.split("\n\n")
    assert paragraphs[0].endswith("[1]")
    assert paragraphs[1].endswith("[1]")


def test_normalize_supported_piln_adds_cited_abstract_definition() -> None:
    evidence = (
        "We proposed a self-supervised image-loop neural network (ILNet) with a part-based model. "
        "The part-based model divides image features into different parts to facilitate "
        "finer-grained learning and improve image details."
    )
    out = finalize_runtime._normalize_citation_plan_supported_terms(
        "ILNet 会把半成品图像送入下一轮迭代。 [1]",
        prompt="ILNet 为什么叫 image-loop，part-based 解决什么？",
        citation_plan={
            "slots": [
                {
                    "preferred_system": "system_a",
                    "source_path": "piln.en.md",
                    "heading_path": "Abstract",
                    "evidence_quote": evidence,
                }
            ]
        },
        answer_hits=[{"meta": {"source_path": "piln.en.md"}}],
    )

    compound = next(paragraph for paragraph in out.split("\n\n") if "self-supervised" in paragraph)
    assert "part-based model" in compound
    assert "finer-grained learning" in compound
    assert compound.endswith("[1]")


def test_normalize_fdm_comparison_uses_the_fdm_hit_not_the_first_plan_hit() -> None:
    fdm_evidence = (
        "Here, we implement frequency-division methods to parallelize the single-pixel "
        "imaging process. The technique enables a trade-off between signal-to-noise ratio "
        "and acquisition speed without altering detector integration time."
    )
    video_evidence = (
        "Photometric stereo senses reflected light with four spatially-separated "
        "single-pixel detectors and reconstructs 3D video at 8 frames per second."
    )
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [4],
                "source_path": "video.en.md",
                "evidence_quote": video_evidence,
            },
            {
                "preferred_system": "system_a",
                "candidate_hits": [1],
                "source_path": "Frequency-division-multiplexed single-pixel imaging.en.md",
                "evidence_quote": fdm_evidence,
            },
        ]
    }
    hits = [
        {
            "text": fdm_evidence,
            "meta": {"source_path": "Frequency-division-multiplexed single-pixel imaging.en.md"},
        },
        {"text": "Other evidence", "meta": {"source_path": "other-1.en.md"}},
        {"text": "Other evidence", "meta": {"source_path": "other-2.en.md"}},
        {"text": video_evidence, "meta": {"source_path": "video.en.md"}},
    ]
    answer = (
        "\u9891\u5206\u590d\u7528\u5b9e\u73b0\u4e86\u56db\u500d\u6548\u7387\u63d0\u5347\uff0c\u5e76\u4e14\u5bf9\u4efb\u610f\u56fe\u50cf\u5c3a\u5bf8\u90fd\u53ef\u6269\u5c55 [3]\u3002\n\n"
        "**3D single-pixel video** \u7528\u56db\u4e2a\u63a2\u6d4b\u5668\u505a photometric stereo [4]\u3002"
    )

    out = finalize_runtime._normalize_citation_plan_supported_terms(
        answer,
        prompt="\u9891\u5206\u590d\u7528\u548c 3D single-pixel video \u5206\u522b\u5e76\u884c\u5316\u4e86\u4ec0\u4e48？",
        citation_plan=plan,
        answer_hits=hits,
    )

    fdm_paragraph = out.split("\n\n")[0]
    assert "[1]" in fdm_paragraph
    assert "[4]" not in fdm_paragraph
    assert "[4]" in out.split("\n\n")[1]


def test_normalize_does_not_treat_generic_lockin_evidence_as_fdm() -> None:
    evidence = (
        "The instrument uses phase-sensitive detection while several frequencies are "
        "measured simultaneously and the signal is demodulated. It can parallelize the "
        "single-pixel imaging process with a trade-off between signal-to-noise ratio and "
        "acquisition speed without altering detector integration time."
    )
    answer = "The experiment reports an unsupported fourfold image-size improvement [3]."

    out = finalize_runtime._normalize_citation_plan_supported_terms(
        answer,
        prompt="Why is FDM faster?",
        citation_plan={
            "slots": [
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [1],
                    "source_path": "Lock-in spectrometer.en.md",
                    "evidence_quote": evidence,
                }
            ]
        },
        answer_hits=[{"text": evidence, "meta": {"source_path": "Lock-in spectrometer.en.md"}}],
    )

    assert out == answer


def test_normalize_supported_terms_completes_existing_microscopy_method_segments() -> None:
    structured_evidence = (
        "Structured detection provides super-resolution, high signal-to-noise ratio, "
        "and enhanced optical sectioning. Since super-resolution and optical sectioning "
        "are achieved simultaneously, we named our technique s2ISM."
    )
    light_field_evidence = (
        "Light-field microscopy is a 3D microscopy technique whereby volumetric information "
        "is gained in a single shot by simultaneously capturing both position and angular "
        "information of light emanating from a sample."
    )
    plan = {
        "budget": {"system_a": 3, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": "iism/iism.en.md",
                "evidence_quote": "Interferometric detection achieves 120 nm lateral resolution.",
            },
            {
                "preferred_system": "system_a",
                "source_path": "light/light.en.md",
                "evidence_quote": light_field_evidence,
            },
            {
                "preferred_system": "system_a",
                "source_path": "structured/structured.en.md",
                "evidence_quote": structured_evidence,
            },
        ],
    }
    hits = [
        {"text": "Interferometric detection evidence.", "meta": {"source_path": "iism/iism.en.md"}},
        {"text": structured_evidence, "meta": {"source_path": "structured/structured.en.md"}},
        {"text": light_field_evidence, "meta": {"source_path": "light/light.en.md"}},
    ]
    answer = (
        "1. **Structured detection（结构化检测，如 ISM）**：解决共聚焦显微中分辨率与信噪比的矛盾 [2]。\n\n"
        "2. **Interferometric（干涉检测，如 iISM）**：降低活细胞成像的光损伤 [1]。\n\n"
        "3. **Light-field（光场显微，LFM）**：缓解空间分辨率与景深之间的取舍 [3]。"
    )

    out = finalize_runtime._normalize_citation_plan_supported_terms(
        answer,
        prompt="structured detection、interferometric、light-field 分别解决什么问题？",
        citation_plan=plan,
        answer_hits=hits,
    )

    structured = next(part for part in out.split("\n\n") if "Structured detection" in part)
    light_field = next(part for part in out.split("\n\n") if "Light-field" in part)
    assert all(
        term in structured
        for term in ("s²ISM", "super-resolution", "optical sectioning", "SNR", "[2]")
    )
    assert all(
        term in light_field
        for term in ("position", "angular information", "volumetric reconstruction", "[3]")
    )
    assert "降低活细胞成像的光损伤 [1]" in out
    assert (
        finalize_runtime._normalize_citation_plan_supported_terms(
            out,
            prompt="structured detection、interferometric、light-field 分别解决什么问题？",
            citation_plan=plan,
            answer_hits=hits,
        )
        == out
    )


def test_method_bundle_completion_preserves_english_answer_language() -> None:
    evidence = (
        "Light-field microscopy obtains volumetric information in a single shot by "
        "simultaneously capturing both position and angular information."
    )
    out = finalize_runtime._complete_grounded_method_bundle_claims(
        "Light-field microscopy addresses the depth-of-field trade-off [1].",
        citation_plan={
            "slots": [
                {
                    "preferred_system": "system_a",
                    "source_path": "light.en.md",
                    "evidence_quote": evidence,
                }
            ]
        },
        answer_hits=[{"text": evidence, "meta": {"source_path": "light.en.md"}}],
    )

    assert "captures both position and angular information" in out
    assert "volumetric reconstruction [1]" in out
    assert "该路线" not in out


def test_method_bundle_completion_restores_light_field_name_in_chinese_answer() -> None:
    evidence = (
        "Light-field microscopy obtains volumetric information in a single shot by "
        "simultaneously capturing both position and angular information."
    )
    out = finalize_runtime._complete_grounded_method_bundle_claims(
        "光场显微镜（LFM）同时记录 position 与 angular information [1]。",
        citation_plan={
            "slots": [
                {
                    "preferred_system": "system_a",
                    "source_path": "light.en.md",
                    "evidence_quote": evidence,
                }
            ]
        },
        answer_hits=[{"text": evidence, "meta": {"source_path": "light.en.md"}}],
    )

    assert "Light-field microscopy（光场显微，LFM）" in out
    assert "position（位置）" in out
    assert "angular information（角度信息）" in out


def test_method_bundle_completion_adds_exact_iism_result_bundle() -> None:
    evidence = (
        "This next-generation technique combines interferometric detection with image "
        "scanning microscopy to achieve about 120 nm lateral resolution while operating "
        "at tenfold lower incident illumination power per diffraction limited spot, "
        "significantly reducing photodamage while enhancing signal-to-noise and contrast."
    )
    answer = "High illumination power can damage live cells."
    plan = {
        "intent": "comparison",
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": "iism.en.md",
                "evidence_quote": evidence,
            }
        ]
    }
    hits = [{"text": evidence, "meta": {"source_path": "iism.en.md"}}]

    out = finalize_runtime._complete_grounded_method_bundle_claims(
        answer,
        citation_plan=plan,
        answer_hits=hits,
    )

    assert "iISM combines interferometric detection" in out
    assert "120 nm lateral resolution" in out
    assert "tenfold lower incident illumination power" in out
    assert "reducing photodamage [1]" in out
    assert (
        finalize_runtime._complete_grounded_method_bundle_claims(
            out,
            citation_plan=plan,
            answer_hits=hits,
        )
        == out
    )


def test_method_bundle_completion_refuses_incomplete_plan_evidence() -> None:
    answer = (
        "Structured detection improves resolution [1].\n\n"
        "Light-field microscopy improves depth of field [2]."
    )
    out = finalize_runtime._complete_grounded_method_bundle_claims(
        answer,
        citation_plan={
            "slots": [
                {
                    "preferred_system": "system_a",
                    "source_path": "structured.en.md",
                    "evidence_quote": "s2ISM provides super-resolution at high SNR.",
                },
                {
                    "preferred_system": "system_a",
                    "source_path": "light.en.md",
                    "evidence_quote": "Light-field microscopy records position for 3D imaging.",
                },
            ]
        },
        answer_hits=[
            {"meta": {"source_path": "structured.en.md"}},
            {"meta": {"source_path": "light.en.md"}},
        ],
    )

    assert out == answer


def test_finalize_keeps_completed_microscopy_method_bundles_grounded(monkeypatch) -> None:
    structured_evidence = (
        "Structured detection provides super-resolution, high signal-to-noise ratio, "
        "and enhanced optical sectioning. Since super-resolution and optical sectioning "
        "are achieved simultaneously, we named our technique s2ISM."
    )
    iism_evidence = (
        "This technique combines interferometric detection with image scanning microscopy "
        "to achieve about 120 nm lateral resolution while operating at tenfold lower "
        "incident illumination power per diffraction limited spot, significantly reducing "
        "photodamage while enhancing signal-to-noise and contrast."
    )
    light_field_evidence = (
        "Light-field microscopy gains volumetric information in a single shot by "
        "simultaneously capturing both position and angular information."
    )
    hits = [
        {"text": iism_evidence, "meta": {"source_path": "iism/iism.en.md"}},
        {"text": structured_evidence, "meta": {"source_path": "structured/structured.en.md"}},
        {"text": light_field_evidence, "meta": {"source_path": "light/light.en.md"}},
    ]
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 3, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_path": "iism/iism.en.md",
                "evidence_quote": iism_evidence,
            },
            {
                "preferred_system": "system_a",
                "source_path": "light/light.en.md",
                "evidence_quote": light_field_evidence,
            },
            {
                "preferred_system": "system_a",
                "source_path": "structured/structured.en.md",
                "evidence_quote": structured_evidence,
            },
        ],
    }
    monkeypatch.setattr(finalize_runtime, "_reconcile_kb_notice", lambda answer, **kwargs: answer)
    monkeypatch.setattr(finalize_runtime, "_enhance_kb_miss_fallback", lambda answer, **kwargs: answer)

    out = finalize_runtime._finalize_generation_answer(
        (
            "1. Structured detection 同时改善分辨率与信噪比 [2]。\n\n"
            "2. 传统高照明功率容易损伤活细胞。\n\n"
            "3. Light-field microscopy 在单次采集中获得体积信息 [3]。"
        ),
        prompt="structured detection、interferometric、light-field 分别解决什么问题？",
        prompt_for_user="structured detection、interferometric、light-field 分别解决什么问题？",
        answer_hits=hits,
        db_dir=Path("db"),
        locked_citation_source=None,
        answer_intent="reading",
        answer_depth="medium",
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
        paper_guide_contracts_seed={"citation_plan": plan},
        apply_paper_guide_answer_postprocess=lambda answer, **kwargs: (answer, []),
        maybe_append_library_figure_markdown=lambda answer, **kwargs: answer,
        validate_structured_citations=lambda answer, **kwargs: (answer, {}),
    )

    assert all(
        term in out["answer"]
        for term in (
            "s²ISM",
            "super-resolution",
            "optical sectioning",
            "iISM",
            "120 nm",
            "入射照明功率",
            "position",
            "angular information",
            "volumetric reconstruction",
        )
    )
    assert out["answer_quality"]["claim_evidence"]["minimum_ok"] is True
    assert out["answer_quality"]["claim_evidence"]["citation_mismatch_claims"] == 0


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


def test_adjacent_duplicate_numeric_citations_are_collapsed() -> None:
    assert finalize_runtime._collapse_adjacent_duplicate_numeric_citations(
        "Evidence [1] [1][1], comparison [2] [2]."
    ) == "Evidence [1], comparison [2]."


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


def test_finalize_binds_planned_source_before_final_claim_audit(monkeypatch) -> None:
    audited_answers: list[str] = []

    def _audit(answer, **_kwargs):
        audited_answers.append(str(answer))
        return str(answer), {"minimum_ok": True}

    monkeypatch.setattr(finalize_runtime, "audit_and_repair_claim_evidence", _audit)
    monkeypatch.setattr(
        finalize_runtime,
        "_reconcile_kb_notice",
        lambda answer, **_kwargs: answer,
    )
    monkeypatch.setattr(
        finalize_runtime,
        "_apply_answer_contract_v1",
        lambda answer, **_kwargs: answer,
    )
    monkeypatch.setattr(
        finalize_runtime,
        "_enhance_kb_miss_fallback",
        lambda answer, **_kwargs: answer,
    )
    monkeypatch.setattr(
        finalize_runtime,
        "_build_answer_quality_probe",
        lambda answer, **_kwargs: {"minimum_ok": True, "answer": answer},
    )
    evidence = (
        "AlphaNet parallelizes coded detector acquisition within one integration time."
    )
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 1, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_name": "AlphaNet",
                "source_path": "alpha-net.en.md",
                "candidate_hits": [1],
                "evidence_quote": evidence,
            }
        ],
    }

    out = finalize_runtime._finalize_generation_answer(
        evidence,
        prompt="How does AlphaNet acquire measurements?",
        prompt_for_user="How does AlphaNet acquire measurements?",
        answer_hits=[
            {
                "text": evidence,
                "meta": {"source_path": "alpha-net.en.md"},
            }
        ],
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
        paper_guide_contracts_seed={"citation_plan": plan},
        apply_paper_guide_answer_postprocess=lambda answer, **_kwargs: (answer, []),
        maybe_append_library_figure_markdown=lambda answer, **_kwargs: answer,
        validate_structured_citations=lambda answer, **_kwargs: (answer, {}),
    )

    assert audited_answers == [
        "AlphaNet parallelizes coded detector acquisition within one integration time [1]."
    ]
    assert audited_answers[0] in out["answer"]


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


def test_finalize_merges_grounded_planned_system_b_opportunity(monkeypatch):
    planned_evidence = (
        "Facing these challenges, video Snapshot Compressive Imaging (SCI) [42] "
        "technology has been developed."
    )
    citation_plan = {
        "version": 1,
        "intent": "origin_lookup",
        "budget": {"system_a": 1, "system_b": 1},
        "slots": [
            {
                "preferred_system": "system_b",
                "candidate_refs": [42],
                "sid": "sscigs123",
                "source_path": "ICIP-2025-SCIGS.en.md",
                "source_name": "SCIGS",
                "topic": "snapshot compressive imaging",
                "heading_path": "1. Introduction",
                "evidence_quote": planned_evidence,
                "grounding_contract": {"context_marker_verified": True},
            }
        ],
    }
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        finalize_runtime,
        "detect_paper_guide_reference_opportunities",
        lambda **_kwargs: [
            {
                "sid": "sscinerf",
                "ref_num": 50,
                "source_path": "SCINeRF.en.md",
                "evidence_quote": "Video SCI [50] has emerged.",
                "context_marker_verified": True,
            }
        ],
    )
    monkeypatch.setattr(
        finalize_runtime,
        "apply_reference_opportunities_to_answer",
        lambda answer, *, opportunities, **_kwargs: (
            answer,
            seen.update({"opportunities": opportunities}) or {"mode": "none"},
        ),
    )

    def _validate(answer, **kwargs):
        seen["candidate_refs"] = kwargs["paper_guide_candidate_refs_by_source"]
        return answer, {"kept": 1}

    finalize_runtime._finalize_generation_answer(
        "SCI's upstream step is snapshot compressive imaging [[CITE:sscigs123:42]].",
        prompt="How did SCI develop from spectral imaging to 3D scene reconstruction?",
        prompt_for_user="How did SCI develop from spectral imaging to 3D scene reconstruction?",
        answer_hits=[{"text": planned_evidence, "meta": {"source_path": "ICIP-2025-SCIGS.en.md"}}],
        db_dir=Path("db"),
        locked_citation_source=None,
        answer_intent="reading",
        answer_depth="L2",
        answer_output_mode="reading_guide",
        paper_guide_mode=True,
        paper_guide_contract_enabled=True,
        paper_guide_prompt_family="overview",
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
        validate_structured_citations=_validate,
    )

    opportunities = list(seen["opportunities"])
    assert [(item["sid"], item["ref_num"]) for item in opportunities] == [
        ("sscigs123", 42)
    ]
    assert seen["candidate_refs"] == {"ICIP-2025-SCIGS.en.md": [42]}


def test_fast_exact_finalize_binds_planned_source_before_claim_audit(monkeypatch) -> None:
    audited_answers: list[str] = []

    def _audit(answer, **_kwargs):
        audited_answers.append(str(answer))
        return str(answer), {"minimum_ok": True}

    monkeypatch.setattr(finalize_runtime, "audit_and_repair_claim_evidence", _audit)
    monkeypatch.setattr(
        finalize_runtime,
        "_build_answer_quality_probe",
        lambda answer, **_kwargs: {"minimum_ok": True, "answer": answer},
    )
    evidence = "AlphaNet parallelizes coded acquisition within one integration time."
    support = {
        "source_path": "alpha-net.en.md",
        "source_name": "AlphaNet",
        "heading_path": "Methods",
        "evidence_quote": evidence,
        "locate_anchor": evidence,
        "block_id": "blk_alpha",
        "anchor_id": "p_alpha",
    }
    plan = {
        "intent": "evidence_lookup",
        "budget": {"system_a": 1, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "source_name": "AlphaNet",
                "source_path": "alpha-net.en.md",
                "candidate_hits": [1],
                "evidence_quote": evidence,
            }
        ],
    }

    out = finalize_runtime._finalize_fast_exact_generation_answer(
        evidence,
        prompt="How does AlphaNet acquire measurements?",
        prompt_for_user="How does AlphaNet acquire measurements?",
        answer_hits=[
            {
                "text": evidence,
                "meta": {"source_path": "alpha-net.en.md"},
            }
        ],
        db_dir=Path("db"),
        locked_citation_source={
            "sid": "s1234abcd",
            "source_name": "AlphaNet",
            "source_path": "alpha-net.en.md",
        },
        answer_intent="reading",
        answer_depth="L2",
        answer_output_mode="reading_guide",
        paper_guide_prompt_family="method",
        paper_guide_bound_source_path="alpha-net.en.md",
        paper_guide_candidate_refs_by_source={},
        paper_guide_support_slots=[support],
        paper_guide_evidence_cards=[],
        paper_guide_precomputed_support_resolution=[support],
        paper_guide_contracts_seed={"citation_plan": plan},
        paper_guide_retrieval_confidence_hint=None,
        research_answer_plan="",
        validate_structured_citations=lambda answer, **_kwargs: (answer, {}),
    )

    assert audited_answers == [
        "AlphaNet parallelizes coded acquisition within one integration time [1]."
    ]
    assert audited_answers[0] in out["answer"]


def test_finalize_fast_exact_reuses_support_without_full_text_rescan(monkeypatch):
    support = {
        "source_path": "paper.md",
        "heading_path": "2. Related Work",
        "block_id": "blk_admm",
        "anchor_id": "p_admm",
        "locate_anchor": "Most existing methods employ ADMM [4].",
        "page_start": 3,
        "page_end": 3,
        "evidence_selection_reason": "exact_support_preflight",
        "strict_locate": True,
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
    original_snapshot = finalize_runtime._build_paper_guide_contract_snapshot

    def _snapshot_with_stale_system_a(**kwargs):
        snapshot = original_snapshot(**kwargs)
        packet = dict(snapshot.get("render_packet") or {})
        packet["cite_details"] = [
            {
                "num": 1,
                "citation_route": "system_a",
                "source_path": "stale.md",
                "evidence_quote": "Stale broad retrieval evidence.",
            },
            *list(packet.get("cite_details") or []),
        ]
        snapshot["render_packet"] = packet
        return snapshot

    monkeypatch.setattr(
        finalize_runtime,
        "_build_paper_guide_contract_snapshot",
        _snapshot_with_stale_system_a,
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
    assert system_a["source_path"] == "paper.md"
    assert system_a["block_id"] == "blk_admm"
    assert system_a["anchor_id"] == "p_admm"
    assert system_a["page_start"] == 3
    assert system_a["page_end"] == 3
    assert system_a["selection_reason"] == "exact_support_preflight"
    assert system_a["strict_locate"] is True
    timing = out["answer_quality"]["_finalize_timing_ms"]
    assert timing["mode"] == "fast_exact"
    assert timing["total"] >= 0


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


def test_contract_snapshot_drops_stale_render_body_even_when_seed_answer_was_updated():
    final_answer = "PILN 的机制见证据 [1]；综述定位见证据 [2]。"
    contracts = finalize_runtime._build_paper_guide_contract_snapshot(
        paper_guide_mode=False,
        intent_model=None,
        answer_markdown=final_answer,
        final_answer_markdown=final_answer,
        evidence_cards=[],
        candidate_refs_by_source={},
        support_slots=[],
        support_resolution=[],
        needs_supplement=False,
        citation_validation={},
        doc_list_contract=[{"source_path": "db/piln.md", "source_name": "PILN"}],
        paper_guide_contracts_seed={
            "render_packet": {
                "answer_markdown": final_answer,
                "rendered_body": "旧回答把无关论文也列为 PILN 证据 [3](#stale)。",
                "rendered_content": "旧回答把无关论文也列为 PILN 证据 [3](#stale)。",
                "copy_markdown": "旧回答把无关论文也列为 PILN 证据 [3]。",
                "copy_text": "旧回答把无关论文也列为 PILN 证据。",
            }
        },
    )

    packet = contracts["render_packet"]
    assert packet["answer_markdown"] == final_answer
    assert packet["rendered_body"] == ""
    assert packet["rendered_content"] == ""
    assert packet["copy_markdown"] == ""
    assert packet["copy_text"] == ""


def test_contract_snapshot_drops_system_a_details_removed_by_final_evidence_gate():
    contracts = finalize_runtime._build_paper_guide_contract_snapshot(
        paper_guide_mode=False,
        intent_model=None,
        answer_markdown="PILN 的机制见证据 [1]；综述定位见证据 [2]。",
        final_answer_markdown="PILN 的机制见证据 [1]；综述定位见证据 [2]。",
        evidence_cards=[],
        candidate_refs_by_source={},
        support_slots=[],
        support_resolution=[],
        needs_supplement=False,
        citation_validation={},
        doc_list_contract=[{"source_path": "db/piln.md", "source_name": "PILN"}],
        paper_guide_contracts_seed={
            "render_packet": {
                "answer_markdown": "PILN 的机制见证据 [1]；综述定位见证据 [2]。",
                "cite_details": [
                    {"citation_route": "system_a", "answer_hit_num": 1, "source_name": "PILN"},
                    {"citation_route": "system_a", "answer_hit_num": 2, "source_name": "Review"},
                    {"citation_route": "system_a", "answer_hit_num": 3, "source_name": "Stale neighbor"},
                ],
            }
        },
    )

    details = contracts["render_packet"]["cite_details"]
    assert [detail["answer_hit_num"] for detail in details] == [1, 2]


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


def test_supplement_skips_complete_two_section_grounded_answer() -> None:
    answer = (
        "## Learned updates\n\nThe primal and dual proximal operators are learned [1].\n\n"
        "## Initialization\n\nThe pseudo-inverse adds complexity, so zero initialization is used [1]."
    )
    builder_calls: list[bool] = []

    out = finalize_runtime._maybe_append_paper_guide_supplement_block(
        answer,
        paper_guide_mode=True,
        has_hits=True,
        prompt_text="Explain the learned updates and initialization choice.",
        prompt_family="method",
        retrieval_confidence_hint={},
        grounded_answer=answer,
        support_resolution=[
            {"support_ok": True, "segment_text": "The proximal operators are learned."},
            {"support_ok": True, "segment_text": "The pseudo-inverse adds complexity."},
        ],
        build_paper_guide_supplement_lines=lambda **kwargs: builder_calls.append(True) or [
            "Implementation detail: unrelated result."
        ],
    )

    assert out == answer
    assert builder_calls == []


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


def test_finalize_generation_answer_rechecks_claims_after_late_answer_mutation(monkeypatch):
    original_audit = finalize_runtime.audit_and_repair_claim_evidence
    audit_calls = 0

    def _counted_audit(*args, **kwargs):
        nonlocal audit_calls
        audit_calls += 1
        return original_audit(*args, **kwargs)

    monkeypatch.setattr(
        finalize_runtime,
        "audit_and_repair_claim_evidence",
        _counted_audit,
    )
    monkeypatch.setattr(finalize_runtime, "_reconcile_kb_notice", lambda answer, **kwargs: answer)
    monkeypatch.setattr(finalize_runtime, "_enhance_kb_miss_fallback", lambda answer, **kwargs: answer)
    monkeypatch.setattr(
        finalize_runtime,
        "_build_answer_quality_probe",
        lambda answer, **kwargs: {"minimum_ok": True},
    )
    monkeypatch.setattr(
        finalize_runtime,
        "_ensure_requested_source_page",
        lambda answer, **kwargs: (
            f"{answer}\n\n该系统还能把所有场景的重建速度提高十倍。"
        ),
    )

    out = finalize_runtime._finalize_generation_answer(
        "该方法让高分辨率中央凹区域跟踪运动 [1]。",
        prompt="foveated dynamic supersampling 如何分配采样资源？",
        prompt_for_user="foveated dynamic supersampling 如何分配采样资源？",
        answer_hits=[
            {
                "text": "A high-resolution foveal region tracks motion within the scene.",
                "meta": {"source_path": "foveated.en.md"},
            }
        ],
        db_dir="db",
        locked_citation_source=None,
        answer_intent="reading",
        answer_depth="medium",
        answer_output_mode="reading_guide",
        paper_guide_mode=True,
        paper_guide_contract_enabled=False,
        paper_guide_prompt_family="method",
        paper_guide_special_focus_block="",
        paper_guide_focus_source_path="",
        paper_guide_direct_source_path="",
        paper_guide_bound_source_path="foveated.en.md",
        paper_guide_candidate_refs_by_source={},
        paper_guide_support_slots=[],
        paper_guide_evidence_cards=[],
        paper_guide_contracts_seed={
            "citation_plan": {
                "intent": "answer_grounding",
                "budget": {"system_a": 1, "system_b": 0},
                "slots": [
                    {
                        "preferred_system": "system_a",
                        "candidate_hits": [1],
                        "source_path": "foveated.en.md",
                        "evidence_quote": (
                            "A high-resolution foveal region tracks motion within the scene."
                        ),
                    }
                ],
            }
        },
        apply_paper_guide_answer_postprocess=lambda answer, **kwargs: (answer, []),
        maybe_append_library_figure_markdown=lambda answer, **kwargs: answer,
        validate_structured_citations=lambda answer, **kwargs: (answer, {}),
    )

    assert "中央凹区域跟踪运动" in out["answer"]
    assert "提高十倍" not in out["answer"]
    assert out["answer_quality"]["claim_evidence"]["final_gate_applied"] is True
    assert out["answer_quality"]["claim_evidence"]["minimum_ok"] is True
    assert audit_calls == 1
    timing = out["answer_quality"]["_finalize_timing_ms"]
    assert timing["mode"] == "standard"
    assert timing["total"] >= 0
    assert set(timing["stages"]) == {
        "answer_contract",
        "citation_routing",
        "citation_precision",
        "answer_shape",
        "evidence_final_gate",
        "supplement_and_contracts",
        "quality_metadata",
    }


def test_finalize_restores_precise_evidence_term_after_final_gate(monkeypatch) -> None:
    original_audit = finalize_runtime.audit_and_repair_claim_evidence
    audit_calls = 0

    def _drop_first_normalized_clause(answer, *args, **kwargs):
        nonlocal audit_calls
        audit_calls += 1
        if audit_calls == 1:
            paragraphs = [
                paragraph
                for paragraph in str(answer or "").split("\n\n")
                if "论文摘要的关键表述" not in paragraph
            ]
            answer = "\n\n".join(paragraphs)
        return original_audit(answer, *args, **kwargs)

    monkeypatch.setattr(
        finalize_runtime,
        "audit_and_repair_claim_evidence",
        _drop_first_normalized_clause,
    )
    monkeypatch.setattr(finalize_runtime, "_reconcile_kb_notice", lambda answer, **kwargs: answer)
    monkeypatch.setattr(finalize_runtime, "_enhance_kb_miss_fallback", lambda answer, **kwargs: answer)
    monkeypatch.setattr(
        finalize_runtime,
        "_build_answer_quality_probe",
        lambda answer, **kwargs: {"minimum_ok": True},
    )
    evidence = (
        "A high-resolution foveal region tracks motion within the scene, yet unlike a "
        "simple zoom, every frame delivers new spatial information from across the entire "
        "field of view. This strategy accumulates detail of slower regions over several "
        "consecutive frames."
    )

    out = finalize_runtime._finalize_generation_answer(
        "Foveated dynamic supersampling 让高分辨率中央凹区域跟踪运动，并跨帧累积细节 [1]。",
        prompt="foveated dynamic supersampling 如何分配时空采样？",
        prompt_for_user="foveated dynamic supersampling 如何分配时空采样？",
        answer_hits=[
            {
                "text": evidence,
                "meta": {"source_path": "foveated.en.md"},
            }
        ],
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
        paper_guide_contracts_seed={
            "citation_plan": {
                "intent": "comparison",
                "budget": {"system_a": 1, "system_b": 0},
                "slots": [
                    {
                        "preferred_system": "system_a",
                        "candidate_hits": [1],
                        "source_path": "foveated.en.md",
                        "evidence_quote": evidence,
                    }
                ],
            }
        },
        apply_paper_guide_answer_postprocess=lambda answer, **kwargs: (answer, []),
        maybe_append_library_figure_markdown=lambda answer, **kwargs: answer,
        validate_structured_citations=lambda answer, **kwargs: (answer, {}),
    )

    assert "整个视场" in out["answer"]
    assert out["answer_quality"]["claim_evidence"]["post_gate_term_normalization"] is True
    assert audit_calls == 2


def test_finalize_rebinds_surviving_fdm_claim_after_the_first_gate(monkeypatch) -> None:
    audit_calls = 0
    bind_calls = 0

    def _drop_initial_bad_fdm_claim(answer, *args, **kwargs):
        nonlocal audit_calls
        audit_calls += 1
        text = str(answer or "")
        if audit_calls == 1:
            text = "\n\n".join(
                paragraph
                for paragraph in text.split("\n\n")
                if "four named carriers" not in paragraph
            )
        return text, {"minimum_ok": True, "audit_call": audit_calls}

    def _staged_bind(answer, *args, **kwargs):
        nonlocal bind_calls
        bind_calls += 1
        text = str(answer or "")
        if bind_calls == 1:
            # Reproduce the live failure: an earlier same-source marker makes
            # the first binder leave the later valid mechanism untouched.
            return text
        paragraphs = text.split("\n\n")
        for index, paragraph in enumerate(paragraphs):
            if "p frequencies simultaneously" in paragraph and "[1]" not in paragraph:
                paragraphs[index] = f"{paragraph} [1]"
        return "\n\n".join(paragraphs)

    monkeypatch.setattr(
        finalize_runtime,
        "audit_and_repair_claim_evidence",
        _drop_initial_bad_fdm_claim,
    )
    monkeypatch.setattr(
        finalize_runtime,
        "_bind_planned_source_citations",
        _staged_bind,
    )
    monkeypatch.setattr(finalize_runtime, "_reconcile_kb_notice", lambda answer, **kwargs: answer)
    monkeypatch.setattr(finalize_runtime, "_enhance_kb_miss_fallback", lambda answer, **kwargs: answer)
    monkeypatch.setattr(
        finalize_runtime,
        "_build_answer_quality_probe",
        lambda answer, **kwargs: {"minimum_ok": True},
    )
    fdm_evidence = (
        "Each SLM pixel is modulated on p frequencies simultaneously according to the "
        "mask patterns. The signal is demodulated by p lock-in amplifiers using "
        "phase-sensitive detection."
    )
    video_evidence = (
        "Photometric stereo uses four spatially-separated single-pixel detectors and "
        "reconstructs 3D video at 8 frames per second."
    )
    answer = (
        "FDM uses four named carriers f1, f2, f3, and f4 [1].\n\n"
        "Frequency-division multiplexed single-pixel imaging parallelizes the encoding "
        "layer: each SLM pixel carries p frequencies simultaneously and the detector "
        "signal is demodulated.\n\n"
        "3D single-pixel video uses photometric stereo with four detectors [4]."
    )
    hits = [
        {"text": fdm_evidence, "meta": {"source_path": "fdm.en.md"}},
        {"text": "Other evidence", "meta": {"source_path": "other-1.en.md"}},
        {"text": "Other evidence", "meta": {"source_path": "other-2.en.md"}},
        {"text": video_evidence, "meta": {"source_path": "video.en.md"}},
    ]
    plan = {
        "intent": "comparison",
        "budget": {"system_a": 2, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [4],
                "source_path": "video.en.md",
                "evidence_quote": video_evidence,
            },
            {
                "preferred_system": "system_a",
                "candidate_hits": [1],
                "source_path": "fdm.en.md",
                "evidence_quote": fdm_evidence,
            },
        ],
    }

    out = finalize_runtime._finalize_generation_answer(
        answer,
        prompt="Compare FDM parallel encoding with 3D single-pixel video.",
        prompt_for_user="Compare FDM parallel encoding with 3D single-pixel video.",
        answer_hits=hits,
        db_dir="db",
        locked_citation_source=None,
        answer_intent="comparison",
        answer_depth="medium",
        answer_output_mode="comparison",
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
        paper_guide_contracts_seed={"citation_plan": plan},
        apply_paper_guide_answer_postprocess=lambda answer, **kwargs: (answer, []),
        maybe_append_library_figure_markdown=lambda answer, **kwargs: answer,
        validate_structured_citations=lambda answer, **kwargs: (answer, {}),
    )

    assert "four named carriers" not in out["answer"]
    surviving_fdm = next(
        paragraph
        for paragraph in out["answer"].split("\n\n")
        if "p frequencies simultaneously" in paragraph
    )
    assert "[1]" in surviving_fdm
    assert "photometric stereo with four detectors [4]" in out["answer"]
    assert bind_calls == 2
    assert audit_calls == 2
    assert out["answer_quality"]["claim_evidence"]["post_gate_citation_rebinding"] is True


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


def test_color_spi_comparison_restores_source_stated_distortion_challenge() -> None:
    source_path = r"F:\db\LPR\LPR.en.md"
    evidence = (
        "Compared with the gray SPI, the color SPI system may require longer "
        "imaging times, and the unknown color response coefficient can inevitably "
        "lead to color distortion. Recently, the DL algorithms have been introduced "
        "into these strategies, which can significantly mitigate the complexity of "
        "the system and reduce the imaging time."
    )
    citation_plan = {
        "intent": "answer_grounding",
        "budget": {"system_a": 1, "system_b": 0},
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [1],
                "source_path": source_path,
                "evidence_quote": evidence,
            }
        ],
    }
    answer_hits = [
        {
            "text": evidence,
            "meta": {"source_path": source_path},
        }
    ]
    answer = (
        "## 直接回答\n\n"
        "深度学习可以降低彩色 SPI 的系统复杂度和成像时间 [1]。\n\n"
        "### 一、彩色 SPI 相比灰度 SPI 的额外挑战\n\n"
        "1. **成像时间更长**：彩色 SPI 需要更长的成像时间 [1]。\n\n"
        "### 二、深度学习如何改善\n\n"
        "深度学习算法可以降低系统复杂度和成像时间 [1]。"
    )

    normalized = finalize_runtime._normalize_citation_plan_supported_terms(
        answer,
        prompt=(
            "彩色单像素成像相比灰度 SPI 有哪些额外挑战？"
            "深度学习怎样降低系统复杂度和成像时间？"
        ),
        citation_plan=citation_plan,
        answer_hits=answer_hits,
    )

    assert "颜色响应系数未知会导致颜色失真" in normalized
    assert "color response coefficient" in normalized
    assert normalized.count("颜色响应系数未知会导致颜色失真") == 1
    assert normalized.index("颜色响应系数未知会导致颜色失真") < normalized.index(
        "### 二、深度学习如何改善"
    )
    audited, meta = finalize_runtime.audit_and_repair_claim_evidence(
        normalized,
        answer_hits=answer_hits,
        allow_citation_repairs=True,
        prompt="彩色单像素成像相比灰度 SPI 有哪些额外挑战？",
        drop_unsupported_high_risk_claims=True,
        enforce_user_visible_binding=True,
    )
    assert "颜色响应系数未知会导致颜色失真" in audited
    assert "color distortion） [1]" in audited
    assert meta["minimum_ok"] is True


def test_normalizer_completes_source_enumeration_without_inventing_definitions() -> None:
    evidence = (
        "The main parameters of single photon detectors are detection efficiency "
        "(DE), dark count, system dead time, time jitter, and so on."
    )
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [1],
                "source_path": "detector-review.en.md",
                "evidence_quote": evidence,
                "page_start": 10,
            }
        ]
    }
    hits = [{"text": evidence, "meta": {"source_path": "detector-review.en.md"}}]

    normalized = finalize_runtime._normalize_citation_plan_supported_terms(
        "还要关注系统死时间和时间抖动 [1]。",
        prompt="除了探测效率，还必须同时看哪些关键指标？",
        citation_plan=plan,
        answer_hits=hits,
    )
    repeated = finalize_runtime._normalize_citation_plan_supported_terms(
        normalized,
        prompt="除了探测效率，还必须同时看哪些关键指标？",
        citation_plan=plan,
        answer_hits=hits,
    )

    assert "detection efficiency" in normalized
    assert "dark count" in normalized
    assert "system dead time" in normalized
    assert "time jitter" in normalized
    assert normalized.count("Complete source enumeration") == 0
    assert repeated == normalized


def test_normalizer_adds_missing_reported_quantity_from_exact_plan_sentence() -> None:
    evidence = (
        "We use inherent position and angular/momentum correlation. "
        "Since each degree of freedom can be measured on separate cameras, no position "
        "resolution is sacrificed for angular resolution. This has allowed us to achieve "
        r"a DOF that is between 2–5 times larger, at the $5\,\mu\mathrm{m}$ resolution, than other methods."
    )
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [1],
                "source_path": "qclfm.en.md",
                "evidence_quote": evidence,
                "page_start": 3,
            }
        ]
    }
    hits = [{"text": evidence, "meta": {"source_path": "qclfm.en.md"}}]

    normalized = finalize_runtime._normalize_citation_plan_supported_terms(
        "QCLFM 用不同相机分别测量位置和角度自由度 [1]。",
        prompt="QCLFM 为什么能保住位置和角度分辨率？实际报告的景深提升有多大？",
        citation_plan=plan,
        answer_hits=hits,
    )
    repeated = finalize_runtime._normalize_citation_plan_supported_terms(
        normalized,
        prompt="QCLFM 为什么能保住位置和角度分辨率？实际报告的景深提升有多大？",
        citation_plan=plan,
        answer_hits=hits,
    )

    assert "2–5 times larger" in normalized
    assert "5 μm" in normalized
    assert "[1]" in normalized
    assert repeated == normalized


def test_normalizer_adds_two_missing_compact_dataset_quantities() -> None:
    dataset_evidence = (
        "Our dataset SA-1B consists of 11M diverse images and 1.1B high-quality "
        "segmentation masks."
    )
    generation_evidence = (
        "We produced 1.1B masks, 99.1% of which were generated fully automatically."
    )
    plan = {
        "slots": [
            {
                "preferred_system": "system_a",
                "candidate_hits": [1],
                "source_path": "sam.en.md",
                "evidence_quote": dataset_evidence,
                "page_start": 6,
            },
            {
                "preferred_system": "system_a",
                "candidate_hits": [1],
                "source_path": "sam.en.md",
                "evidence_quote": generation_evidence,
                "page_start": 6,
            },
        ]
    }
    hits = [
        {
            "text": f"{dataset_evidence} {generation_evidence}",
            "meta": {"source_path": "sam.en.md"},
        }
    ]

    normalized = finalize_runtime._normalize_citation_plan_supported_terms(
        "SAM 使用三阶段数据引擎 [1]。",
        prompt="SA-1B 有多少图像和掩码，其中多少比例是全自动生成的？",
        citation_plan=plan,
        answer_hits=hits,
    )
    repeated = finalize_runtime._normalize_citation_plan_supported_terms(
        normalized,
        prompt="SA-1B 有多少图像和掩码，其中多少比例是全自动生成的？",
        citation_plan=plan,
        answer_hits=hits,
    )

    assert "11M" in normalized
    assert "1.1B" in normalized
    assert "99.1%" in normalized
    assert normalized.count("原文定量结果：") == 2
    assert repeated == normalized


def test_grounded_fact_completion_keeps_compound_dataset_quantities_in_one_claim() -> None:
    evidence = (
        "Our dataset SA-1B consists of 11M diverse images and 1.1B high-quality "
        "segmentation masks. We produced 1.1B masks, 99.1% of which were "
        "generated fully automatically."
    )
    completed = finalize_runtime._complete_grounded_requested_source_facts(
        "SAM uses a three-stage data engine [1].",
        prompt="How many images and masks are in SA-1B and what share was automatic?",
        citation_plan={
            "slots": [
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [1],
                    "source_path": "sam.en.md",
                    "evidence_quote": evidence,
                    "compound_same_page_evidence": True,
                }
            ]
        },
        answer_hits=[{"text": evidence, "meta": {"source_path": "sam.en.md"}}],
    )

    quantitative_claim = next(
        paragraph for paragraph in completed.split("\n\n") if "Reported quantitative result" in paragraph
    )
    assert all(value in quantitative_claim for value in ("11M", "1.1B", "99.1%"))
    assert quantitative_claim.count("[1]") == 1


def test_grounded_fact_completion_expands_prompt_acronym_on_independent_cited_claims() -> None:
    evidence = (
        "The retrieval component is based on DPR. We use a pre-trained bi-encoder "
        "from DPR to initialize our retriever and build the document index."
    )
    completed = finalize_runtime._complete_grounded_requested_source_facts(
        "Upstream paper: DPR was introduced by Karpukhin et al.\n"
        "RAG's retriever uses DPR's pre-trained bi-encoder [1].",
        prompt=(
            "Did RAG invent Dense Passage Retrieval (DPR), or reuse prior work? "
            "Identify Karpukhin and explain its retriever."
        ),
        citation_plan={
            "slots": [
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [1],
                    "source_path": "rag.en.md",
                    "evidence_quote": evidence,
                }
            ]
        },
        answer_hits=[{"text": evidence, "meta": {"source_path": "rag.en.md"}}],
    )

    assert "Dense Passage Retrieval (DPR) was introduced by Karpukhin" in completed
    assert "retriever uses Dense Passage Retrieval (DPR)'s pre-trained bi-encoder" in completed


def test_origin_system_b_marker_moves_from_experiment_to_upstream_identity() -> None:
    marker = "[[CITE:s20cdc71c:26]]"
    answer = (
        "Upstream paper: DPR comes from Karpukhin et al., Dense Passage Retrieval "
        "for Open-Domain Question Answering.\n\n"
        f"RAG follows DPR's experimental setup for WebQuestions {marker}."
    )

    relocated = finalize_runtime._relocate_planned_origin_system_b_markers(
        answer,
        citation_plan={
            "slots": [
                {
                    "claim_type": "origin",
                    "preferred_system": "system_b",
                    "topic": "DPR",
                    "sid": "s20cdc71c",
                    "candidate_refs": [26],
                }
            ]
        },
    )

    lines = relocated.splitlines()
    assert marker in lines[0]
    assert marker not in lines[-1]
    assert relocated.count(marker) == 1


def test_grounded_fact_completion_restores_missing_ddpm_relation_slots() -> None:
    objective = (
        "3.4 Simplified training objective. The t > 1 cases correspond to an "
        "unweighted version of Eq. (12). L_simple predicts epsilon."
    )
    tradeoff = (
        "4 Experiments / 4.1 Sample quality. Training on the true variational "
        "bound yields better codelengths, but the simplified objective yields "
        "the best sample quality."
    )
    completed = finalize_runtime._complete_grounded_requested_source_facts(
        "DDPM 的 L_simple 让网络预测 epsilon [1]。\n\n样本质量与码长的权衡。",
        prompt=(
            "DDPM 的 L_simple 实际让网络预测什么？它和变分下界的加权有何不同，"
            "论文报告的样本质量与码长权衡是什么？"
        ),
        citation_plan={
            "slots": [
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [1],
                    "source_path": "ddpm.en.md",
                    "evidence_quote": objective,
                },
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [1],
                    "source_path": "ddpm.en.md",
                    "evidence_quote": tradeoff,
                },
            ]
        },
        answer_hits=[{"text": objective, "meta": {"source_path": "ddpm.en.md"}}],
    )

    assert "unweighted version" in completed
    assert "true variational bound" in completed
    assert "better codelengths" in completed
    assert "best sample quality" in completed
    assert completed.count("[1]") >= 3


def test_exact_source_answer_completes_ddpm_objective_and_tradeoff() -> None:
    objective = (
        "3.4 Simplified training objective. The t > 1 cases correspond to an "
        "unweighted version of Eq. (12). L_simple predicts epsilon."
    )
    tradeoff = (
        "We find that training on the true variational bound yields better "
        "codelengths, but the simplified objective yields the best sample quality."
    )
    completed = finalize_runtime._complete_exact_source_bound_answer_claims(
        "Provider returned an incomplete outline.",
        prompt=(
            "DDPM 的 L_simple 实际让网络预测什么？它和变分下界的加权有何不同，"
            "论文报告的样本质量与码长权衡是什么？"
        ),
        citation_plan={
            "slots": [
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [1],
                    "source_path": "ddpm.en.md",
                    "evidence_quote": objective,
                },
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [1],
                    "source_path": "ddpm.en.md",
                    "evidence_quote": tradeoff,
                },
            ]
        },
        answer_hits=[{"text": objective, "meta": {"source_path": "ddpm.en.md"}}],
    )

    assert "L_{\\text{simple}}" in completed
    assert "unweighted version" in completed
    assert "true variational bound" in completed
    assert "better codelengths" in completed
    assert "best sample quality" in completed
    assert completed.count("[1]") == 2


def test_grounded_fact_completion_keeps_same_slot_scale_sentences_one_cited_claim() -> None:
    evidence = (
        "CLIP was trained on 400 million image-text pairs collected from the internet. "
        "This scale exceeds earlier natural-language-supervision datasets."
    )
    completed = finalize_runtime._complete_grounded_requested_source_facts(
        "Regarding pretraining data scale, CLIP was trained on 400 million image-text "
        "pairs collected from the internet [1]. This scale was a key motivation [1].",
        prompt="What was CLIP's pretraining data scale?",
        citation_plan={
            "slots": [
                {
                    "preferred_system": "system_a",
                    "candidate_hits": [1],
                    "source_path": "clip.en.md",
                    "evidence_quote": evidence,
                }
            ]
        },
        answer_hits=[{"text": evidence, "meta": {"source_path": "clip.en.md"}}],
    )

    assert "internet; This scale was a key motivation [1]." in completed
    assert completed.count("[1]") == 1
